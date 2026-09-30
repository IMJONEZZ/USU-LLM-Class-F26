import shutil
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import modal
import pytest

from src.generation import (
    build_image,
    build_test_image,
    generate,
    load_model,
    log_gpu_usage,
)


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


class FakeOutOfMemoryError(Exception):
    pass


def _fake_torch() -> MagicMock:
    fake_torch = MagicMock()
    fake_torch.cuda.OutOfMemoryError = FakeOutOfMemoryError
    return fake_torch


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
        patch(
            "src.generation.load_model", return_value=(fake_model, fake_tokenizer)
        ) as mock_load_model,
        patch.dict(
            sys.modules, {"transformers": fake_transformers, "torch": _fake_torch()}
        ),
    ):
        result = generate.local("unsloth/Llama-3.2-1B")

    mock_load_model.assert_called_once_with("unsloth/Llama-3.2-1B")
    fake_transformers.set_seed.assert_called_once_with(42)
    fake_tokenizer.assert_called_once_with("Once upon a time", return_tensors="pt")
    fake_encoded.to.assert_called_once_with("cuda")
    fake_model.generate.assert_called_once_with(
        input_ids="fake_ids", temperature=1, do_sample=True, max_new_tokens=50
    )
    fake_tokenizer.decode.assert_called_once_with(
        "fake_output_row", skip_special_tokens=True
    )
    assert result is True

    captured = capsys.readouterr()
    assert "generated text" in captured.out


def test_generate_returns_false_on_out_of_memory(capsys):
    def raise_oom(model_name):
        raise FakeOutOfMemoryError("CUDA out of memory")

    with (
        patch("src.generation.load_model", side_effect=raise_oom),
        patch.dict(sys.modules, {"transformers": MagicMock(), "torch": _fake_torch()}),
    ):
        result = generate.local("unsloth/Llama-3.2-70B")

    assert result is False
    captured = capsys.readouterr()
    assert "unsloth/Llama-3.2-70B" in captured.out
    assert "T4" in captured.out


def test_log_gpu_usage_prints_percentage(capsys):
    stop_event = threading.Event()
    fake_result = MagicMock()
    fake_result.stdout = "512, 15360\n"

    def fake_run(*args, **kwargs):
        stop_event.set()
        return fake_result

    with patch("src.generation.subprocess.run", side_effect=fake_run) as mock_run:
        log_gpu_usage(stop_event, interval=5.0)

    mock_run.assert_called_once_with(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    captured = capsys.readouterr()
    assert "512/15360 MiB (3.3%)" in captured.out


@pytest.mark.skipif(
    shutil.which("nvidia-smi") is None, reason="requires a real GPU with nvidia-smi"
)
def test_log_gpu_usage_real_gpu():
    stop_event = threading.Event()
    monitor = threading.Thread(target=log_gpu_usage, args=(stop_event, 5.0))
    monitor.start()
    time.sleep(11)
    stop_event.set()
    monitor.join()
