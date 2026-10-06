from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch
from transformers import EarlyStoppingCallback, TrainerControl, TrainerState

from src.prompts import INSTRUCTION_MARKER, RESPONSE_MARKER, format_example
from src.trainer import (
    build_text_dataset,
    build_trainer,
    load_model_for_training,
    make_early_stopping_callback,
)


def make_fake_unsloth():
    fake_unsloth = MagicMock()
    fake_unsloth.FastLanguageModel.from_pretrained.return_value = (
        "base_model",
        "tokenizer",
    )
    fake_unsloth.FastLanguageModel.get_peft_model.return_value = "peft_model"
    return fake_unsloth


def load_with_fake_unsloth():
    fake_unsloth = make_fake_unsloth()
    with patch.dict("sys.modules", {"unsloth": fake_unsloth}):
        result = load_model_for_training()
    return fake_unsloth.FastLanguageModel, result


def test_load_model_for_training_loads_the_base_model_in_4bit():
    fast_language_model, _ = load_with_fake_unsloth()

    kwargs = fast_language_model.from_pretrained.call_args.kwargs
    assert kwargs["model_name"] == "unsloth/Llama-3.2-1B"
    assert kwargs["load_in_4bit"] is True
    assert kwargs["max_seq_length"] == 8192


def test_load_model_for_training_attaches_qlora_adapters_with_unsloth_defaults():
    fast_language_model, _ = load_with_fake_unsloth()

    args = fast_language_model.get_peft_model.call_args.args
    kwargs = fast_language_model.get_peft_model.call_args.kwargs
    assert args == ("base_model",)
    assert kwargs["r"] == 16
    assert kwargs["lora_alpha"] == 16
    assert kwargs["lora_dropout"] == 0
    assert kwargs["bias"] == "none"
    assert kwargs["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def test_load_model_for_training_turns_on_unsloth_gradient_checkpointing():
    fast_language_model, _ = load_with_fake_unsloth()

    kwargs = fast_language_model.get_peft_model.call_args.kwargs
    assert kwargs["use_gradient_checkpointing"] == "unsloth"


def test_load_model_for_training_returns_the_peft_model_and_tokenizer():
    _, result = load_with_fake_unsloth()
    assert result == ("peft_model", "tokenizer")


def test_load_model_for_training_does_not_switch_to_inference_mode():
    fast_language_model, _ = load_with_fake_unsloth()
    fast_language_model.for_inference.assert_not_called()


def test_build_text_dataset_formats_each_resume_and_ends_with_the_eos_token():
    resumes = pd.DataFrame({"Resume_str": ["r1", "r2"], "Category": ["HR", "BPO"]})
    categories = ["BPO", "HR"]

    dataset = build_text_dataset(resumes, categories, "<eos>")

    assert list(dataset["text"]) == [
        format_example("r1", "HR", categories) + "<eos>",
        format_example("r2", "BPO", categories) + "<eos>",
    ]


EARLY_STOPPING_ARGS = SimpleNamespace(
    metric_for_best_model="eval_loss", greater_is_better=False
)


def test_early_stopping_callback_stops_when_eval_loss_worsens():
    callback = make_early_stopping_callback()
    state = TrainerState(best_metric=1.0)
    control = TrainerControl()

    callback.on_evaluate(
        EARLY_STOPPING_ARGS, state, control, metrics={"eval_loss": 1.2}
    )

    assert control.should_training_stop is True


def test_early_stopping_callback_keeps_training_while_eval_loss_improves():
    callback = make_early_stopping_callback()
    state = TrainerState(best_metric=1.0)
    control = TrainerControl()

    callback.on_evaluate(
        EARLY_STOPPING_ARGS, state, control, metrics={"eval_loss": 0.8}
    )

    assert control.should_training_stop is False


def build_trainer_with_fakes():
    fake_trl = MagicMock()
    fake_chat_templates = MagicMock()
    fake_chat_templates.train_on_responses_only.return_value = "masked_trainer"
    fake_unsloth = MagicMock()
    fake_unsloth.chat_templates = fake_chat_templates
    fake_modules = {
        "trl": fake_trl,
        "unsloth": fake_unsloth,
        "unsloth.chat_templates": fake_chat_templates,
    }
    with patch.dict("sys.modules", fake_modules):
        result = build_trainer("model", "tokenizer", "train_data", "eval_data", "out")
    return fake_trl, fake_chat_templates, result


def test_build_trainer_configures_guide_defaults_and_evaluates_every_10_steps():
    fake_trl, _, _ = build_trainer_with_fakes()

    kwargs = fake_trl.SFTConfig.call_args.kwargs
    assert kwargs["output_dir"] == "out"
    assert kwargs["per_device_train_batch_size"] == 2
    assert kwargs["gradient_accumulation_steps"] == 8
    assert kwargs["learning_rate"] == 2e-4
    assert kwargs["max_steps"] == 30
    assert kwargs["eval_strategy"] == "steps"
    assert kwargs["eval_steps"] == 10
    assert kwargs["save_strategy"] == "steps"
    assert kwargs["save_steps"] == 10
    assert kwargs["load_best_model_at_end"] is True
    assert kwargs["metric_for_best_model"] == "eval_loss"
    assert kwargs["dataset_text_field"] == "text"


def test_build_trainer_does_not_truncate_below_the_longest_example():
    fake_trl, _, _ = build_trainer_with_fakes()

    assert fake_trl.SFTConfig.call_args.kwargs["max_length"] == 8192


def test_build_trainer_passes_model_data_and_early_stopping_to_sft_trainer():
    fake_trl, _, _ = build_trainer_with_fakes()

    kwargs = fake_trl.SFTTrainer.call_args.kwargs
    assert kwargs["model"] == "model"
    assert kwargs["processing_class"] == "tokenizer"
    assert kwargs["train_dataset"] == "train_data"
    assert kwargs["eval_dataset"] == "eval_data"
    assert kwargs["args"] is fake_trl.SFTConfig.return_value
    assert [type(callback) for callback in kwargs["callbacks"]] == [
        EarlyStoppingCallback
    ]
    assert kwargs["callbacks"][0].early_stopping_patience == 1


def test_build_trainer_masks_everything_before_the_response():
    fake_trl, fake_chat_templates, result = build_trainer_with_fakes()

    fake_chat_templates.train_on_responses_only.assert_called_once_with(
        fake_trl.SFTTrainer.return_value,
        instruction_part=INSTRUCTION_MARKER,
        response_part=RESPONSE_MARKER,
    )
    assert result == "masked_trainer"


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_masking_leaves_only_the_category_and_one_eos_token_unmasked(tmp_path):
    model, tokenizer = load_model_for_training()
    categories = ["BPO", "HR"]
    resumes = pd.DataFrame(
        {
            "Resume_str": ["Managed payroll and hiring.", "Answered customer calls."],
            "Category": ["HR", "BPO"],
        }
    )
    dataset = build_text_dataset(resumes, categories, tokenizer.eos_token)
    trainer = build_trainer(model, tokenizer, dataset, dataset, str(tmp_path))

    batch = next(iter(trainer.get_train_dataloader()))

    # Unsloth packs the batch into one row (padding-free), so check the whole batch
    eos = tokenizer.eos_token
    unmasked_ids = batch["input_ids"][batch["labels"] != -100]
    assert tokenizer.decode(unmasked_ids) in {f"HR{eos}BPO{eos}", f"BPO{eos}HR{eos}"}
