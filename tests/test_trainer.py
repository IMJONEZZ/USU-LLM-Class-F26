"""Exercise the trainer export without downloads, GPU use, or model training."""

import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest


def chat_generator(answers):
    answers = iter(answers)

    def respond(messages, **kwargs):
        return [
            {
                "generated_text": [
                    *messages,
                    {"role": "assistant", "content": next(answers)},
                ]
            }
        ]

    return Mock(side_effect=respond)


class TinyDataset:
    """Apply the script's map/filter callbacks to actual small examples."""

    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0]) if rows else []

    def map(self, function, *, fn_kwargs, remove_columns):
        assert remove_columns == self.column_names
        return TinyDataset([function(row, **fn_kwargs) for row in self.rows])

    def filter(self, predicate):
        return TinyDataset([row for row in self.rows if predicate(row)])

    def train_test_split(self, *, test_size, seed):
        # A deterministic stand-in for the library split, not a test of its randomness.
        assert 0 < test_size < 1
        assert isinstance(seed, int)
        return {"train": self.rows[:-1], "test": self.rows[-1:]}


@pytest.fixture(params=[False, True], ids=["cpu", "cuda"])
def workflow(request, monkeypatch, capsys):
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
    rows = [
        {"input": "First riddle", "expected_answer": "Echo"},
        {"input": "Second riddle", "expected_answer": "Map"},
        {"input": "x" * 600, "expected_answer": "Too long"},
    ]
    evaluation_data = pd.DataFrame(rows[:2], index=[9, 2])
    loader = Mock(
        side_effect=lambda name, split: (
            TinyDataset(rows) if split == "train" else evaluation_data
        )
    )
    module("datasets", load_dataset=loader)
    model = object()
    model_loader = Mock(return_value=model)
    trained_model = object()
    trainer = Mock(model=model)
    events = Mock()

    def train():
        trainer.model = trained_model

    trainer.train.side_effect = train
    events.attach_mock(trainer.train, "train")
    events.attach_mock(trainer.save_model, "save_model")
    events.attach_mock(tokenizer.save_pretrained, "save_tokenizer")
    trainer_constructor = Mock(return_value=trainer)
    collator = Mock()
    collator_constructor = Mock(return_value=collator)
    args_constructor = Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
    before = chat_generator(["Map", "Echo"])
    after = chat_generator(["Echo", "Globe"])
    pipeline = Mock(side_effect=[before, after])
    utils = module("transformers.utils", logging=Mock())
    module(
        "transformers",
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_loader),
        DataCollatorForSeq2Seq=collator_constructor,
        Trainer=trainer_constructor,
        TrainingArguments=args_constructor,
        pipeline=pipeline,
        utils=utils,
    )
    vectors = {"Echo": [1, 0], "Map": [0, 1], "Globe": [3, 4]}
    embedder = Mock()
    embedder.encode.side_effect = lambda texts: np.array(
        [vectors[text] for text in texts]
    )
    embedder_constructor = Mock(return_value=embedder)
    module("sentence_transformers", SentenceTransformer=embedder_constructor)
    module("torch", cuda=Mock(is_available=Mock(return_value=request.param)))
    module("tqdm", tqdm=lambda rows, **kwargs: rows)
    path = Path(__file__).resolve().parents[1] / "src" / "trainer.py"
    namespace = runpy.run_path(str(path), run_name="__main__")
    printed = capsys.readouterr().out
    return SimpleNamespace(**locals())


def test_workflow_masks_data_trains_saves_and_evaluates(workflow):
    w = workflow
    ns = w.namespace
    w.tokenizer_loader.assert_called_once_with(ns["model_name"])
    assert w.tokenizer.pad_token == ("<pad>" if w.request.param else "<end>")
    assert w.tokenizer.padding_side == "right"
    assert [call.kwargs["split"] for call in w.loader.call_args_list] == [
        "train",
        "test",
    ]
    prepared = ns["dataset"]["train"] + ns["dataset"]["test"]
    assert len(prepared) == 2  # The overlong row was actually filtered out.
    for row, answer in zip(prepared, ["Echo", "Map"], strict=True):
        assert (
            "".join(chr(token) for token in row["labels"] if token != -100)
            == answer + "<end>"
        )
        assert row["labels"][0] == -100
    w.collator_constructor.assert_called_once_with(w.tokenizer, label_pad_token_id=-100)
    kwargs = w.trainer_constructor.call_args.kwargs
    assert kwargs["train_dataset"] is ns["dataset"]["train"]
    assert kwargs["eval_dataset"] is ns["dataset"]["test"]
    assert kwargs["data_collator"] is w.collator
    assert kwargs["processing_class"] is w.tokenizer
    assert kwargs["model"] is w.model
    assert [call[0] for call in w.events.mock_calls] == [
        "train",
        "save_model",
        "save_tokenizer",
    ]
    w.trainer.save_model.assert_called_once_with(ns["training_args"].output_dir)
    w.tokenizer.save_pretrained.assert_called_once_with(ns["training_args"].output_dir)
    assert w.pipeline.call_args_list[0].kwargs["model"] == ns["model_name"]
    assert w.pipeline.call_args_list[1].kwargs["model"] is w.trained_model
    for call in w.pipeline.call_args_list:
        assert call.kwargs["device"] == (0 if w.request.param else -1)
    w.embedder_constructor.assert_called_once_with(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cuda" if w.request.param else "cpu",
    )
    assert ns["before_results"][0] == pytest.approx(0)
    assert ns["after_results"][0] == pytest.approx(0.9)
    for generator in [w.before, w.after]:
        for call, riddle in zip(
            generator.call_args_list, w.evaluation_data["input"], strict=True
        ):
            assert call.args[0] == [
                {"role": "system", "content": ns["prompt_base"]},
                {"role": "user", "content": riddle},
            ]
            assert call.kwargs == {"do_sample": False, "max_new_tokens": 16}
    assert "Mean similarity after fine-tuning:" in w.printed


@pytest.mark.parametrize("max_length", [5, 6])
def test_tokenize_preserves_answer_and_end_token_at_length_boundary(
    workflow, max_length
):
    tokenizer = Mock()
    tokenizer.apply_chat_template.side_effect = [[10, 11], [10, 11, 20, 21, 99]]
    example = {"input": "Riddle?", "expected_answer": "Two tokens"}
    original = example.copy()
    result = workflow.namespace["tokenize_riddle"](
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
        {"role": "user", "content": "Riddle?"},
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
    result = workflow.namespace["tokenize_riddle"](
        {"input": "Riddle", "expected_answer": "Answer"},
        tokenizer,
        "Instruction",
        max_length=4,
    )
    assert result == {"input_ids": [], "attention_mask": [], "labels": []}


def test_tokenize_rejects_inconsistent_chat_prefix(workflow):
    tokenizer = Mock()
    tokenizer.apply_chat_template.side_effect = [[10, 11], [10, 12, 20, 99]]
    with pytest.raises(ValueError, match="chat template changed the prompt prefix"):
        workflow.namespace["tokenize_riddle"](
            {"input": "Riddle", "expected_answer": "Answer"}, tokenizer, "Instruction"
        )


@pytest.mark.parametrize(
    ("vectors", "score"),
    [
        ([[3, 4], [6, 8]], 1.0),
        ([[2, 0], [0, 3]], 0.0),
        ([[3, 4], [-6, -8]], -1.0),
        ([[1, 0], [3, 4]], 0.6),
    ],
)
def test_evaluate_similarity_output_and_input_preservation(workflow, vectors, score):
    data = pd.DataFrame(
        {"input": ["Riddle"], "expected_answer": ["Reference"]}, index=[7]
    )
    original = data.copy(deep=True)
    generator = chat_generator(["Answer: a prediction"])
    embedder = Mock()
    embedder.encode.return_value = np.array(vectors)
    mean, scores, output = workflow.namespace["evaluate"](
        generator, data, embedder, "Custom instruction"
    )
    assert mean == pytest.approx(score)
    assert scores == pytest.approx([score])
    assert output["similarity"].tolist() == pytest.approx([score])
    pd.testing.assert_frame_equal(
        output.drop(columns="similarity"),
        data.assign(generated_answer="Answer: a prediction"),
    )
    pd.testing.assert_frame_equal(data, original)
    embedder.encode.assert_called_once_with(["Reference", "Answer: a prediction"])
    assert generator.call_args.args[0] == [
        {"role": "system", "content": "Custom instruction"},
        {"role": "user", "content": "Riddle"},
    ]
