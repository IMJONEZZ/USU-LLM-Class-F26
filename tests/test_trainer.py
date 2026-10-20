import importlib
import sys
from types import ModuleType, SimpleNamespace

from src import trainer
from src.trainer import tokenize_example


class FakeTokenizer:
    eos_token_id = 99

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict:
        if add_special_tokens:
            return {"input_ids": [10, 11, 12]}
        return {"input_ids": [20, 21]}


def test_tokenize_example_masks_prompt_and_keeps_output_labels() -> None:
    result = tokenize_example(
        {"question": "turn the valve", "answer": "**TURN** the valve."},
        FakeTokenizer(),
        max_length=8,
    )

    assert result["input_ids"] == [10, 11, 12, 20, 21, 99]
    assert result["labels"] == [-100, -100, -100, 20, 21, 99]
    assert result["attention_mask"] == [1, 1, 1, 1, 1, 1]


def test_tokenize_example_truncates_prompt_to_reserve_output_tokens() -> None:
    result = tokenize_example(
        {"question": "a long instruction", "answer": "response"},
        FakeTokenizer(),
        max_length=4,
    )

    assert result["input_ids"] == [12, 20, 21, 99]
    assert result["labels"] == [-100, 20, 21, 99]
    assert len(result["input_ids"]) <= 4


def test_train_model_runs_pipeline_with_mocked_dependencies(monkeypatch) -> None:
    calls = {}

    class FakeTokenizerForTraining(FakeTokenizer):
        pad_token = None
        eos_token = "<eos>"
        pad_token_id = 99

        def save_pretrained(self, path):
            calls["tokenizer_saved_to"] = path

    class FakeDataset:
        @property
        def column_names(self):
            return ["question", "answer"]

        def train_test_split(self, **kwargs):
            calls["split_kwargs"] = kwargs
            return FakeSplits()

    class FakeSplits(dict):
        def __init__(self):
            super().__init__(train=[{"question": "a", "answer": "b"}], test=[])

        def map(self, fn, *, remove_columns):
            calls["remove_columns"] = remove_columns
            calls["tokenized_example"] = fn({"question": "a", "answer": "b"})
            return self

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace()

        def print_trainable_parameters(self):
            calls["printed_parameters"] = True

    class FakeTrainer:
        def __init__(self, **kwargs):
            calls["trainer_kwargs"] = kwargs

        def train(self):
            return SimpleNamespace(metrics={"train_loss": 0.25})

        def evaluate(self):
            return {"eval_loss": 0.5}

        def save_model(self, path):
            calls["model_saved_to"] = path

    fake_tokenizer = FakeTokenizerForTraining()
    fake_model = FakeModel()
    fake_hf_api = SimpleNamespace(
        whoami=lambda: {"name": "test-user"},
        create_repo=lambda **kwargs: calls.setdefault("created_repo", kwargs),
        upload_folder=lambda **kwargs: calls.setdefault("uploaded_folder", kwargs),
    )

    def fake_module(name, **attrs):
        module = ModuleType(name)
        for attr, value in attrs.items():
            setattr(module, attr, value)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    fake_module("datasets", load_dataset=lambda *args, **kwargs: FakeDataset())
    fake_module("huggingface_hub", HfApi=lambda token: fake_hf_api)
    fake_module(
        "peft",
        LoraConfig=lambda **kwargs: kwargs,
        TaskType=SimpleNamespace(CAUSAL_LM="CAUSAL_LM"),
        get_peft_model=lambda model, _config: model,
    )
    fake_module(
        "transformers",
        AutoModelForCausalLM=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: fake_model
        ),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda _model_id: fake_tokenizer),
        DataCollatorForSeq2Seq=lambda **kwargs: kwargs,
        Trainer=FakeTrainer,
        TrainingArguments=lambda **kwargs: kwargs,
    )
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(trainer, "modal", object())
    monkeypatch.setattr(
        trainer,
        "output_volume",
        SimpleNamespace(commit=lambda: calls.update(volume_committed=True)),
        raising=False,
    )

    result = trainer._train_model()

    assert calls["split_kwargs"] == {"test_size": 0.2, "seed": 42}
    assert calls["remove_columns"] == ["question", "answer"]
    assert calls["tokenized_example"]["labels"][-1] == 99
    assert fake_tokenizer.pad_token == "<eos>"
    assert fake_tokenizer.padding_side == "right"
    assert fake_model.config.use_cache is False
    assert calls["model_saved_to"] == "/outputs/lora_adapter"
    assert calls["tokenizer_saved_to"] == "/outputs/lora_adapter"
    assert calls["volume_committed"] is True
    assert calls["created_repo"]["repo_id"] == "test-user/A6-llama-3.2-1b-lora"
    assert calls["uploaded_folder"]["folder_path"] == "/outputs/lora_adapter"
    assert result == {
        "repo_id": "test-user/A6-llama-3.2-1b-lora",
        "train_metrics": {"train_loss": 0.25},
        "eval_metrics": {"eval_loss": 0.5},
    }


def test_modal_training_app_registration(monkeypatch) -> None:
    calls = {}

    class App:
        def function(self, **kwargs):
            calls["function_kwargs"] = kwargs
            return lambda fn: fn

    class Image:
        def pip_install(self, *packages):
            calls["packages"] = packages
            return self

        def add_local_file(self, local_path, *, remote_path):
            calls["local_file"] = local_path
            calls["remote_path"] = remote_path
            return self

    class Volume:
        @staticmethod
        def from_name(name, *, create_if_missing):
            calls["volume"] = (name, create_if_missing)
            return "volume"

    fake_modal = SimpleNamespace(
        App=lambda name: calls.update(app=name) or App(),
        Volume=Volume,
        Secret=SimpleNamespace(
            from_local_environ=lambda names: calls.update(secret=names) or "secret"
        ),
        Image=SimpleNamespace(debian_slim=lambda: Image()),
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setenv("HF_TOKEN", "secret")

    reloaded = importlib.reload(trainer)

    assert calls["app"] == "gpu-test"
    assert calls["volume"] == ("A6-lora-output", True)
    assert calls["secret"] == ["HF_TOKEN"]
    assert calls["remote_path"] == "/training_data/data_1.csv"
    assert calls["function_kwargs"]["gpu"] == "T4"
    assert calls["function_kwargs"]["volumes"] == {"/outputs": "volume"}
    assert calls["function_kwargs"]["secrets"] == ["secret"]
    assert isinstance(calls["function_kwargs"]["image"], Image)
    assert reloaded.train_model is reloaded._train_model
