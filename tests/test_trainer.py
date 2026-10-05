from unittest.mock import MagicMock, patch

from src.trainer import load_model_for_training


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
    assert kwargs["max_seq_length"] == 4096


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
