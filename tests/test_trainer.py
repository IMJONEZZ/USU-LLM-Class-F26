"""Unit tests for the mocked LoRA training orchestration."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src import trainer as trainer_module


@pytest.fixture
def training_mocks(monkeypatch, tmp_path):
    """Replace model, dataset, and Trainer APIs with small local fixtures."""
    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<eos>"
    tokenizer.pad_token_id = 7

    def save_tokenizer(output_dir):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    tokenizer.save_pretrained.side_effect = save_tokenizer

    base_model = MagicMock()
    base_model.config = SimpleNamespace(pad_token_id=None)
    lora_model = MagicMock()
    lora_model.config = SimpleNamespace(pad_token_id=None)

    fast_language_model = SimpleNamespace(
        from_pretrained=MagicMock(return_value=(base_model, tokenizer)),
        get_peft_model=MagicMock(return_value=lora_model),
    )
    monkeypatch.setitem(
        sys.modules,
        "unsloth",
        SimpleNamespace(FastLanguageModel=fast_language_model),
    )

    # Two short examples stand in for tokenized dialogue shorter than block_size.
    datasets = {
        "train": [{"input_ids": [1, 2, 3]}],
        "validation": [{"input_ids": [4, 5]}],
    }
    create_datasets = MagicMock(return_value=datasets)
    monkeypatch.setattr(trainer_module, "create_datasets", create_datasets)

    trainer = MagicMock()
    trainer.evaluate.side_effect = [
        {"eval_loss": 4.0},
        {"eval_loss": 2.5},
    ]
    trainer.train.return_value = SimpleNamespace(metrics={"train_runtime": 12.5})

    def save_adapter(output_dir):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "adapter_config.json").write_text("{}", encoding="utf-8")
        (output_path / "adapter_model.safetensors").write_bytes(b"adapter")

    trainer.save_model.side_effect = save_adapter

    training_arguments = MagicMock(side_effect=lambda **kwargs: kwargs)
    data_collator = MagicMock(return_value="collator")
    trainer_factory = MagicMock(return_value=trainer)
    transformers_module = SimpleNamespace(
        DataCollatorForLanguageModeling=data_collator,
        Trainer=trainer_factory,
        TrainingArguments=training_arguments,
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    return SimpleNamespace(
        tokenizer=tokenizer,
        base_model=base_model,
        lora_model=lora_model,
        fast_language_model=fast_language_model,
        datasets=datasets,
        create_datasets=create_datasets,
        trainer=trainer,
        trainer_factory=trainer_factory,
        transformers=transformers_module,
        training_arguments=training_arguments,
        data_collator=data_collator,
        tmp_path=tmp_path,
        torch=torch,
    )


def test_run_training_uses_defaults_saves_artifacts_and_returns_metrics(
    training_mocks, monkeypatch
):
    monkeypatch.chdir(training_mocks.tmp_path)

    metrics = trainer_module.run_training()

    output_dir = training_mocks.tmp_path / "models/star-wars-llama"
    assert (output_dir / "adapter_config.json").is_file()
    assert (output_dir / "adapter_model.safetensors").is_file()
    assert (output_dir / "tokenizer_config.json").is_file()
    training_mocks.trainer.save_model.assert_called_once_with("models/star-wars-llama")
    training_mocks.tokenizer.save_pretrained.assert_called_once_with(
        "models/star-wars-llama"
    )

    assert metrics == {
        "training_runtime_seconds": 12.5,
        "initial_validation_loss": 4.0,
        "final_validation_loss": 2.5,
        "validation_loss_improvement": 1.5,
        "validation_loss_improvement_percent": 37.5,
        "peak_vram_allocated_gib": None,
        "peak_vram_reserved_gib": None,
    }

    training_mocks.fast_language_model.from_pretrained.assert_called_once_with(
        model_name=trainer_module.MODEL_NAME,
        max_seq_length=256,
        dtype=None,
        load_in_4bit=True,
    )
    training_mocks.create_datasets.assert_called_once_with(
        training_mocks.tokenizer, block_size=256
    )
    assert (
        training_mocks.trainer_factory.call_args.kwargs["train_dataset"]
        == (training_mocks.datasets["train"])
    )
    assert (
        training_mocks.trainer_factory.call_args.kwargs["eval_dataset"]
        == (training_mocks.datasets["validation"])
    )
    assert training_mocks.training_arguments.call_args.kwargs["output_dir"] == (
        "models/star-wars-llama"
    )
    assert training_mocks.training_arguments.call_args.kwargs["num_train_epochs"] == 3


def test_run_training_uses_eos_token_when_tokenizer_has_no_padding_token(
    training_mocks,
):
    trainer_module.run_training(output_dir=str(training_mocks.tmp_path / "output"))

    assert training_mocks.tokenizer.pad_token == training_mocks.tokenizer.eos_token
    assert training_mocks.lora_model.config.pad_token_id == (
        training_mocks.tokenizer.pad_token_id
    )


def test_run_training_converts_peak_cuda_memory_to_gib(training_mocks, monkeypatch):
    torch = training_mocks.torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", MagicMock())
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", MagicMock())
    monkeypatch.setattr(torch.cuda, "synchronize", MagicMock())
    monkeypatch.setattr(
        torch.cuda, "max_memory_allocated", MagicMock(return_value=2 * 1024**3)
    )
    monkeypatch.setattr(
        torch.cuda, "max_memory_reserved", MagicMock(return_value=3 * 1024**3)
    )

    metrics = trainer_module.run_training(
        output_dir=str(training_mocks.tmp_path / "output")
    )

    assert metrics["peak_vram_allocated_gib"] == 2.0
    assert metrics["peak_vram_reserved_gib"] == 3.0
