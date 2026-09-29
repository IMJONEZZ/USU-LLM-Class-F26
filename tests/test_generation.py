import sys
from unittest.mock import MagicMock, patch

import modal

from src.generation import build_image, build_test_image, generate, load_model


def test_build_image_returns_modal_image():
    image = build_image()

    assert isinstance(image, modal.Image)


def test_build_test_image_returns_modal_image():
    image = build_test_image()

    assert isinstance(image, modal.Image)


def test_load_model_calls_from_pretrained_and_for_inference():
    fake_model = MagicMock(name="model")
    fake_tokenizer = MagicMock(name="tokenizer")

    fake_fast_language_model = MagicMock()
    fake_fast_language_model.from_pretrained.return_value = (fake_model, fake_tokenizer)

    fake_unsloth = MagicMock()
    fake_unsloth.FastLanguageModel = fake_fast_language_model

    with patch.dict(sys.modules, {"unsloth": fake_unsloth}):
        model, tokenizer = load_model()

    fake_fast_language_model.from_pretrained.assert_called_once_with(
        model_name="unsloth/Llama-3.2-1B", load_in_16bit=True, load_in_4bit=False
    )

    fake_fast_language_model.for_inference.assert_called_once_with(fake_model)
    assert model is fake_model
    assert tokenizer is fake_tokenizer


def test_generate_calls_expected_pipeline(capsys):
    fake_encoded = MagicMock()
    fake_encoded.to.return_value = {"input_ids": "fake_ids"}

    fake_tokenizer = MagicMock()
    fake_tokenizer.return_value = fake_encoded
    fake_tokenizer.decode.return_value = "generated text"

    fake_model = MagicMock()
    fake_model.generate.return_value = ["fake_output_row"]

    fake_transformers = MagicMock()

    with (
        patch("src.generation.load_model", return_value=(fake_model, fake_tokenizer)),
        patch.dict(sys.modules, {"transformers": fake_transformers}),
    ):
        generate.local()

    fake_transformers.set_seed.assert_called_once_with(42)
    fake_tokenizer.assert_called_once_with("Once upon a time", return_tensors="pt")
    fake_encoded.to.assert_called_once_with("cuda")
    fake_model.generate.assert_called_once_with(
        input_ids="fake_ids", temperature=1, do_sample=True
    )
    fake_tokenizer.decode.assert_called_once_with(
        "fake_output_row", skip_special_tokens=True
    )

    captured = capsys.readouterr()
    assert "generated text" in captured.out
