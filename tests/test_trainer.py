"""Exercise the trainer export without downloads, GPU use, or model training."""

import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


def chat_generator(answer):
    def respond(messages, **kwargs):
        return [
            {"generated_text": [*messages, {"role": "assistant", "content": answer}]}
        ]

    return Mock(side_effect=respond)


class TinyDataset:
    """Run preprocessing callbacks on small examples, without Hugging Face."""

    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0]) if rows else []

    @classmethod
    def from_list(cls, rows):
        return cls(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return self.rows[key]

    def map(self, function, *, remove_columns, fn_kwargs=None):
        assert remove_columns == self.column_names
        return TinyDataset([function(row, **(fn_kwargs or {})) for row in self.rows])

    def filter(self, predicate):
        return TinyDataset([row for row in self.rows if predicate(row)])

    def train_test_split(self, *, test_size, seed):
        assert 0 < test_size < 1
        assert isinstance(seed, int)
        return {
            "train": TinyDataset(self.rows[:-1]),
            "test": TinyDataset(self.rows[-1:]),
        }


@pytest.fixture(params=[False, True], ids=["cpu", "cuda"])
def workflow_builder(request, monkeypatch, capsys):
    def run(*, articles=None, rows=None):
        def module(name, **attributes):
            stub = ModuleType(name)
            stub.__dict__.update(attributes)
            monkeypatch.setitem(sys.modules, name, stub)
            return stub

        tokenizer = Mock()
        tokenizer.pad_token = "<pad>" if request.param else None
        tokenizer.eos_token = "<end>"

        def template(messages, *, tokenize, add_generation_prompt, return_dict):
            assert tokenize is True
            assert return_dict is False
            text = "".join(f"<{m['role']}>{m['content']}<end>" for m in messages)
            if add_generation_prompt:
                text += "<assistant>"
            return [ord(character) for character in text]

        tokenizer.apply_chat_template.side_effect = template
        tokenizer_loader = Mock(return_value=tokenizer)
        if rows is None:
            rows = [
                {
                    "topic": "Lateral thinking",
                    "problem_statement": "First puzzle",
                    "solution": "Explain first. Answer: Echo",
                },
                {
                    "topic": "Math problems",
                    "problem_statement": "Excluded puzzle",
                    "solution": "Excluded",
                },
                {
                    "topic": "Lateral thinking",
                    "problem_statement": "Second puzzle",
                    "solution": "Explain second. Answer: Map",
                },
                {
                    "topic": "Lateral thinking",
                    "problem_statement": "x" * 2200,
                    "solution": "Too long",
                },
            ]
        loader = Mock(return_value=TinyDataset(rows))
        module("datasets", Dataset=TinyDataset, load_dataset=loader)
        base_model, adapter_model, wikipedia_model, final_model = [
            Mock() for _ in range(4)
        ]
        model_loader = Mock(return_value=base_model)
        lora_constructor = Mock(return_value=object())
        get_peft_model = Mock(return_value=adapter_model)
        module("peft", LoraConfig=lora_constructor, get_peft_model=get_peft_model)
        wikipedia_trainer = Mock(model=adapter_model)
        trainer = Mock(model=wikipedia_model)
        wikipedia_trainer.train.side_effect = lambda: setattr(
            wikipedia_trainer, "model", wikipedia_model
        )
        trainer.train.side_effect = lambda: setattr(trainer, "model", final_model)
        events = Mock()
        events.attach_mock(wikipedia_trainer.train, "wikipedia_train")
        events.attach_mock(wikipedia_trainer.save_model, "wikipedia_save")
        events.attach_mock(trainer.train, "lateral_train")
        events.attach_mock(trainer.save_model, "lateral_save")
        events.attach_mock(tokenizer.save_pretrained, "save_tokenizer")
        trainer_constructor = Mock(side_effect=[wikipedia_trainer, trainer])
        collator = Mock()
        collator_constructor = Mock(return_value=collator)
        args_constructor = Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
        stopping_constructor = Mock(return_value=object())
        before, after = (
            chat_generator("Original answer"),
            chat_generator("Trained answer"),
        )
        pipeline = Mock(side_effect=[before, after])
        utils = module("transformers.utils", logging=Mock())
        module(
            "transformers",
            AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_loader),
            DataCollatorForSeq2Seq=collator_constructor,
            Trainer=trainer_constructor,
            TrainingArguments=args_constructor,
            EarlyStoppingCallback=stopping_constructor,
            pipeline=pipeline,
            utils=utils,
        )
        module("sentence_transformers", SentenceTransformer=Mock())
        module("torch", cuda=Mock(is_available=Mock(return_value=request.param)))
        module("tqdm", tqdm=lambda rows, **kwargs: rows)
        root = Path(__file__).resolve().parents[1]
        path = root / "src" / "trainer.py"
        data_dir = root / "data"
        if articles is None:
            articles = {"unit_topic.txt": "A complete article about the topic."}
        article_paths = {data_dir / name: content for name, content in articles.items()}
        real_glob, real_read_text = Path.glob, Path.read_text

        def glob(directory, pattern, *args, **kwargs):
            if directory == data_dir and pattern == "*.txt":
                return iter(article_paths)
            return real_glob(directory, pattern, *args, **kwargs)

        def read_text(file, *args, **kwargs):
            if file in article_paths:
                return article_paths[file]
            return real_read_text(file, *args, **kwargs)

        monkeypatch.setattr(Path, "glob", glob)
        monkeypatch.setattr(Path, "read_text", read_text)
        namespace = runpy.run_path(str(path), run_name="__main__")
        printed = capsys.readouterr().out
        return SimpleNamespace(**locals(), cuda=request.param)

    return run


@pytest.fixture
def workflow(workflow_builder):
    return workflow_builder()


def test_workflow_prepares_solutions_and_masks_prompts(workflow):
    w = workflow
    ns = w.namespace
    w.loader.assert_called_once_with("hivaze/LOGIC-701", "en", split="train")
    w.tokenizer_loader.assert_called_once_with(ns["model_name"])
    assert w.tokenizer.pad_token == ("<pad>" if w.cuda else "<end>")
    assert w.tokenizer.padding_side == "right"
    prepared = ns["dataset"]["train"].rows + ns["dataset"]["test"].rows
    assert len(prepared) == 2  # Non-lateral and overlong examples are excluded.
    for row, answer in zip(
        prepared,
        ["Explain first. Answer: Echo", "Explain second. Answer: Map"],
        strict=True,
    ):
        assert "".join(chr(t) for t in row["labels"] if t != -100) == answer + "<end>"
        assert row["labels"][0] == -100
        assert len(row["input_ids"]) == len(row["attention_mask"]) == len(row["labels"])
    article = ns["wikipedia_dataset"][0]
    assert (
        "".join(chr(t) for t in article["labels"] if t != -100)
        == "A complete article about the topic.<end>"
    )
    assert "Describe unit topic." in "".join(map(chr, article["input_ids"]))


def test_workflow_chains_adapter_training_and_saves_each_stage(workflow):
    w = workflow
    ns = w.namespace
    w.get_peft_model.assert_called_once_with(
        w.base_model, w.lora_constructor.return_value
    )
    assert w.lora_constructor.call_args.kwargs["task_type"] == "CAUSAL_LM"
    w.adapter_model.print_trainable_parameters.assert_called_once()
    w.collator_constructor.assert_called_once_with(w.tokenizer, label_pad_token_id=-100)
    first, second = [call.kwargs for call in w.trainer_constructor.call_args_list]
    assert first["model"] is w.adapter_model
    assert first["train_dataset"] is ns["wikipedia_dataset"]
    assert second["model"] is w.wikipedia_model
    assert second["train_dataset"] is ns["dataset"]["train"]
    assert second["eval_dataset"] is ns["dataset"]["test"]
    for kwargs in (first, second):
        assert kwargs["data_collator"] is w.collator
        assert kwargs["processing_class"] is w.tokenizer
    assert [call[0] for call in w.events.mock_calls] == [
        "wikipedia_train",
        "wikipedia_save",
        "save_tokenizer",
        "lateral_train",
        "lateral_save",
        "save_tokenizer",
    ]
    assert ns["wikipedia_training_args"].output_dir != ns["training_args"].output_dir
    w.wikipedia_trainer.save_model.assert_called_once_with(
        ns["wikipedia_training_args"].output_dir
    )
    w.trainer.save_model.assert_called_once_with(ns["training_args"].output_dir)
    assert [call.args[0] for call in w.tokenizer.save_pretrained.call_args_list] == [
        ns["wikipedia_training_args"].output_dir,
        ns["training_args"].output_dir,
    ]
    assert second["callbacks"] == [w.stopping_constructor.return_value]
    w.stopping_constructor.assert_called_once_with(
        early_stopping_patience=5, early_stopping_threshold=0.0
    )


def test_generators_compare_original_and_final_models(workflow):
    w = workflow
    ns = w.namespace
    assert w.pipeline.call_args_list[0].kwargs["model"] == ns["model_name"]
    assert w.pipeline.call_args_list[1].kwargs["model"] is w.final_model
    w.final_model.eval.assert_called_once()
    for call in w.pipeline.call_args_list:
        assert call.kwargs["device"] == (0 if w.cuda else -1)
    for generator in (w.before, w.after):
        assert generator.call_count == len(ns["final_questions"])
        for call, question in zip(
            generator.call_args_list, ns["final_questions"], strict=True
        ):
            assert call.args[0] == [
                {"role": "system", "content": ns["prompt_base"]},
                {"role": "user", "content": question},
            ]
            assert call.kwargs["do_sample"] is False
    assert "Answer after fine tuning: Trained answer" in w.printed


@pytest.mark.parametrize("max_length", [5, 6])
def test_tokenize_preserves_answer_and_end_token_at_length_boundary(
    workflow, max_length
):
    tokenizer = Mock()
    tokenizer.apply_chat_template.side_effect = [[10, 11], [10, 11, 20, 21, 99]]
    example = {"input": "Puzzle?", "expected_answer": "Two tokens"}
    original = example.copy()
    result = workflow.namespace["tokenize_question"](
        example, tokenizer, "Instruction", max_length
    )
    assert result == {
        "input_ids": [10, 11, 20, 21, 99],
        "attention_mask": [1, 1, 1, 1, 1],
        "labels": [-100, -100, 20, 21, 99],
    }
    assert example == original
    calls = tokenizer.apply_chat_template.call_args_list
    assert calls[0].args[0] == [
        {"role": "system", "content": "Instruction"},
        {"role": "user", "content": "Puzzle?"},
    ]
    assert calls[0].kwargs["add_generation_prompt"] is True
    assert calls[1].args[0] == [
        *calls[0].args[0],
        {"role": "assistant", "content": "Two tokens"},
    ]
    assert calls[1].kwargs["add_generation_prompt"] is False


def test_tokenize_skips_overlong_conversation_without_partial_answer(workflow):
    tokenizer = Mock()
    tokenizer.apply_chat_template.side_effect = [[10, 11], [10, 11, 20, 21, 99]]
    result = workflow.namespace["tokenize_question"](
        {"input": "Puzzle", "expected_answer": "Answer"},
        tokenizer,
        "Instruction",
        max_length=4,
    )
    assert result == {"input_ids": [], "attention_mask": [], "labels": []}


def test_tokenize_rejects_inconsistent_chat_prefix(workflow):
    tokenizer = Mock()
    tokenizer.apply_chat_template.side_effect = [[10, 11], [10, 12, 20, 99]]
    with pytest.raises(ValueError, match="chat template changed the prompt prefix"):
        workflow.namespace["tokenize_question"](
            {"input": "Puzzle", "expected_answer": "Answer"},
            tokenizer,
            "Instruction",
        )


@pytest.mark.parametrize(
    ("articles", "message"),
    [
        ({}, "No Wikipedia"),
        ({"empty.txt": "  \n"}, "file is empty"),
        ({"long.txt": "x" * 4200}, "exceeds 4096"),
    ],
)
def test_wikipedia_rejects_missing_empty_or_overlong_articles(
    workflow_builder, articles, message
):
    with pytest.raises(ValueError, match=message):
        workflow_builder(articles=articles)


def test_lateral_thinking_requires_multiple_examples(workflow_builder):
    with pytest.raises(ValueError, match="at least two labeled"):
        workflow_builder(
            rows=[
                {
                    "topic": "Lateral thinking",
                    "problem_statement": "Puzzle",
                    "solution": "Answer",
                }
            ]
        )


def test_lateral_thinking_rechecks_size_after_length_filter(workflow_builder):
    rows = [
        {
            "topic": "Lateral thinking",
            "problem_statement": "x" * 2200,
            "solution": "Answer",
        }
    ] * 2
    with pytest.raises(ValueError, match="Too few lateral-thinking"):
        workflow_builder(rows=rows)
