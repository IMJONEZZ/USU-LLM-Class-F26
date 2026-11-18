"""Tests for loading and combining the trainer's question-answer data."""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import pytest

from src import trainer


@pytest.fixture
def mocked_training_runtime(monkeypatch, tmp_path):
    """Provide fake ML libraries so the Modal training body runs on the CPU."""
    state = SimpleNamespace(
        formatted_rows=None,
        trainer=None,
        committed=False,
        model_saved=False,
        tokenizer_saved=False,
        response_masking=None,
    )

    class FakeModel:
        def save_pretrained(self, directory):
            Path(directory).mkdir(parents=True, exist_ok=True)
            state.model_saved = True

    class FakeTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is False
            return f"formatted: {messages[0]['content']} => {messages[1]['content']}"

        def save_pretrained(self, directory):
            assert Path(directory).is_dir()
            state.tokenizer_saved = True

    model = FakeModel()
    tokenizer = FakeTokenizer()

    class FakeFastLanguageModel:
        @staticmethod
        def from_pretrained(**kwargs):
            assert kwargs["model_name"] == trainer.MODEL_ID
            return model, tokenizer

        @staticmethod
        def get_peft_model(model_arg, **kwargs):
            assert model_arg is model
            assert kwargs["r"] == 16
            return model_arg

    class FakeDataset:
        @classmethod
        def from_list(cls, rows):
            state.formatted_rows = rows
            return rows

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            self.values = kwargs

    class FakeSFTTrainer:
        def __init__(self, **kwargs):
            state.trainer = self
            self.values = kwargs
            self.trained = False

        def train(self):
            self.trained = True

    unsloth_module = ModuleType("unsloth")
    unsloth_module.__path__ = []
    unsloth_module.FastLanguageModel = FakeFastLanguageModel
    chat_templates_module = ModuleType("unsloth.chat_templates")

    def fake_train_on_responses_only(sft_trainer, **kwargs):
        state.response_masking = kwargs
        return sft_trainer

    chat_templates_module.train_on_responses_only = fake_train_on_responses_only
    torch_module = ModuleType("torch")
    torch_module.cuda = SimpleNamespace(is_bf16_supported=lambda: False)
    datasets_module = ModuleType("datasets")
    datasets_module.Dataset = FakeDataset
    transformers_module = ModuleType("transformers")
    transformers_module.TrainingArguments = FakeTrainingArguments
    trl_module = ModuleType("trl")
    trl_module.SFTTrainer = FakeSFTTrainer

    for name, module in {
        "unsloth": unsloth_module,
        "unsloth.chat_templates": chat_templates_module,
        "torch": torch_module,
        "datasets": datasets_module,
        "transformers": transformers_module,
        "trl": trl_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setattr(trainer, "OUTPUT_MOUNT_PATH", str(tmp_path / "outputs"))
    monkeypatch.setattr(
        trainer,
        "output_volume",
        SimpleNamespace(commit=lambda: setattr(state, "committed", True)),
    )
    monkeypatch.setenv("HF_TOKEN", "test-token")
    return state, tmp_path / "outputs"


def test_load_manual_examples_skips_blank_lines_and_strips_text(tmp_path):
    data_path = tmp_path / "manual.jsonl"
    data_path.write_text(
        '\n{"question":"  What is this?  ","answer":"  An example.  "}\n',
        encoding="utf-8",
    )

    examples = trainer.load_manual_examples(data_path)

    assert examples == [
        {
            "task_type": "manual",
            "messages": [
                {"role": "user", "content": "What is this?"},
                {"role": "assistant", "content": "An example."},
            ],
        }
    ]


def test_load_manual_examples_requires_existing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Manual examples file not found"):
        trainer.load_manual_examples(tmp_path / "missing.jsonl")


def test_load_manual_examples_reports_invalid_json_line(tmp_path):
    data_path = tmp_path / "manual.jsonl"
    data_path.write_text('{"question":\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"Invalid JSON.*line 1"):
        trainer.load_manual_examples(data_path)


@pytest.mark.parametrize(
    ("record", "message", "exception_type"),
    [
        ([], "must be a JSON object", TypeError),
        ({"answer": "Answer."}, "Missing question", ValueError),
        ({"question": "Question?"}, "Missing answer", ValueError),
        ({"question": "  ", "answer": "Answer."}, "Missing question", ValueError),
        ({"question": "Question?", "answer": "  "}, "Missing answer", ValueError),
    ],
)
def test_load_manual_examples_validates_records(
    tmp_path, record, message, exception_type
):
    data_path = tmp_path / "manual.jsonl"
    data_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(exception_type, match=message):
        trainer.load_manual_examples(data_path)


def test_load_training_examples_combines_truthfulqa_and_manual(monkeypatch, tmp_path):
    rows = [
        {"Question": "  First question? ", "Best Answer": " First answer. "},
        {"Question": "Second question?", "Best Answer": "Second answer."},
    ]

    # A small class gives the fake dataset normal iteration behavior.
    class FakeDataset:
        column_names: ClassVar[list[str]] = ["Question", "Best Answer"]

        def __iter__(self):
            return iter(rows)

    def load_dataset(dataset_id, split):
        return FakeDataset()

    monkeypatch.setattr(trainer, "load_dataset", load_dataset)

    manual_path = tmp_path / "manual.jsonl"
    manual_path.write_text(
        '{"question":"Manual question?","answer":"Manual answer."}\n',
        encoding="utf-8",
    )

    examples = trainer.load_training_examples(
        dataset_id="example/dataset",
        row_limit=2,
        manual_dataset_path=manual_path,
    )

    assert len(examples) == 3
    assert examples[0] == {
        "task_type": "truthfulqa",
        "messages": [
            {"role": "user", "content": "First question?"},
            {"role": "assistant", "content": "First answer."},
        ],
    }
    assert examples[2]["task_type"] == "manual"
    assert examples[2]["messages"][0]["content"] == "Manual question?"


def test_load_training_examples_requests_first_n_rows(monkeypatch, tmp_path):
    calls = []

    class FakeDataset:
        column_names: ClassVar[list[str]] = ["Question", "Best Answer"]

        def __iter__(self):
            return iter(
                [
                    {"Question": "Question?", "Best Answer": "Answer."},
                ]
            )

    def fake_load_dataset(dataset_id, split):
        calls.append((dataset_id, split))
        return FakeDataset()

    monkeypatch.setattr(trainer, "load_dataset", fake_load_dataset)
    manual_path = tmp_path / "manual.jsonl"
    manual_path.write_text("", encoding="utf-8")

    trainer.load_training_examples(
        dataset_id="example/dataset",
        row_limit=1,
        manual_dataset_path=manual_path,
    )

    assert calls == [("example/dataset", "train[:1]")]


@pytest.mark.parametrize("row_limit", [0, -1])
def test_load_training_examples_rejects_nonpositive_row_limit(monkeypatch, row_limit):
    def unexpected_load(*args, **kwargs):
        pytest.fail("Dataset loading should not run with an invalid row limit")

    monkeypatch.setattr(trainer, "load_dataset", unexpected_load)

    with pytest.raises(ValueError, match="row_limit must be at least 1"):
        trainer.load_training_examples(row_limit=row_limit)


def test_load_training_examples_rejects_invalid_training_row_limit(monkeypatch):
    monkeypatch.setattr(trainer, "TRAINING_ROW_LIMIT", 0)

    with pytest.raises(ValueError, match="row_limit must be at least 1"):
        trainer.load_training_examples()


def test_load_training_examples_reports_missing_columns(monkeypatch):
    class FakeDataset:
        column_names: ClassVar[list[str]] = ["Question"]

    def fake_load_dataset(*args, **kwargs):
        return FakeDataset()

    monkeypatch.setattr(trainer, "load_dataset", fake_load_dataset)

    with pytest.raises(ValueError, match="missing required columns.*Best Answer"):
        trainer.load_training_examples(row_limit=1)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"Question": " ", "Best Answer": "Answer."}, "Missing question"),
        ({"Question": "Question?", "Best Answer": " "}, "Missing Best Answer"),
    ],
)
def test_load_training_examples_validates_dataset_rows(monkeypatch, row, message):
    class FakeDataset:
        column_names: ClassVar[list[str]] = ["Question", "Best Answer"]

        def __iter__(self):
            return iter([row])

    def fake_load_dataset(*args, **kwargs):
        return FakeDataset()

    monkeypatch.setattr(trainer, "load_dataset", fake_load_dataset)

    with pytest.raises(ValueError, match=message):
        trainer.load_training_examples(row_limit=1)


def test_load_training_examples_rejects_short_dataset(monkeypatch):
    class FakeDataset:
        column_names: ClassVar[list[str]] = ["Question", "Best Answer"]

        def __iter__(self):
            return iter([])

    def fake_load_dataset(*args, **kwargs):
        return FakeDataset()

    monkeypatch.setattr(trainer, "load_dataset", fake_load_dataset)

    with pytest.raises(ValueError, match="Expected 1 rows.*got 0"):
        trainer.load_training_examples(row_limit=1)


def test_load_training_examples_rejects_limit_above_available_rows(
    monkeypatch, tmp_path
):
    class FakeDataset:
        column_names: ClassVar[list[str]] = ["Question", "Best Answer"]

        def __iter__(self):
            return iter([{"Question": "Only row?", "Best Answer": "One answer."}])

    def fake_load_dataset(*args, **kwargs):
        return FakeDataset()

    monkeypatch.setattr(trainer, "load_dataset", fake_load_dataset)
    manual_path = tmp_path / "manual.jsonl"
    manual_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected 2 rows.*got 1"):
        trainer.load_training_examples(
            row_limit=2,
            manual_dataset_path=manual_path,
        )


def test_load_training_examples_uses_only_manual_source(monkeypatch, tmp_path):
    def unexpected_load(*args, **kwargs):
        pytest.fail("TruthfulQA should not load when dataset_id is None")

    monkeypatch.setattr(trainer, "load_dataset", unexpected_load)
    manual_path = tmp_path / "manual.jsonl"
    manual_path.write_text(
        '{"question":"Manual question?","answer":"Manual answer."}\n',
        encoding="utf-8",
    )

    examples = trainer.load_training_examples(
        dataset_id=None,
        manual_dataset_path=manual_path,
    )

    assert len(examples) == 1
    assert examples[0]["task_type"] == "manual"
    assert examples[0]["messages"][1]["content"] == "Manual answer."


def test_load_training_examples_uses_only_dataset_source(monkeypatch):
    class FakeDataset:
        column_names: ClassVar[list[str]] = ["Question", "Best Answer"]

        def __iter__(self):
            return iter([{"Question": "Dataset question?", "Best Answer": "Answer."}])

    def fake_load_dataset(dataset_id, split):
        assert dataset_id == "example/dataset"
        assert split == "train[:1]"
        return FakeDataset()

    monkeypatch.setattr(trainer, "load_dataset", fake_load_dataset)

    examples = trainer.load_training_examples(
        dataset_id="example/dataset",
        row_limit=1,
        manual_dataset_path=None,
    )

    assert len(examples) == 1
    assert examples[0]["task_type"] == "truthfulqa"
    assert examples[0]["messages"][0]["content"] == "Dataset question?"


def test_load_training_examples_requires_at_least_one_source():
    with pytest.raises(ValueError, match="At least one training data source"):
        trainer.load_training_examples(dataset_id=None, manual_dataset_path=None)


def test_train_lora_formats_trains_and_saves_adapter(mocked_training_runtime):
    state, output_path = mocked_training_runtime
    examples = [
        {
            "messages": [
                {"role": "user", "content": "Question?"},
                {"role": "assistant", "content": "Answer."},
            ]
        }
    ]

    result = trainer.train_lora.local(examples, "test-run")

    run_path = output_path / "test-run"
    adapter_path = run_path / "adapter"
    assert state.formatted_rows == [{"text": "formatted: Question? => Answer."}]
    assert state.trainer.trained is True
    assert state.trainer.values["dataset_text_field"] == "text"
    assert state.response_masking == {
        "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
        "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    }
    assert state.model_saved is True
    assert state.tokenizer_saved is True
    assert state.committed is True
    assert adapter_path.is_dir()
    manifest = json.loads((run_path / "training_manifest.json").read_text())
    assert manifest["model"] == trainer.MODEL_ID
    assert manifest["training_examples"] == 1
    assert manifest["adapter_path"] == str(adapter_path)
    assert result == f"Saved LoRA adapter to {adapter_path}"


def test_train_lora_requires_huggingface_token(mocked_training_runtime, monkeypatch):
    monkeypatch.delenv("HF_TOKEN")

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        trainer.train_lora.local([], "test-run")
