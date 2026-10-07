"""Offline script checks plus a tiny real generation test when CUDA is available.

The mocked tests check control flow, not GPU correctness or Llama model quality.
runpy executes the script without changing its command-line behavior.
"""

import atexit
import runpy
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@pytest.fixture
def script_environment(monkeypatch):
    # Never inherit a user's model, token, or long inspection pause.
    for key, value in {
        "MODEL_ID": "test-local-model",
        "PROMPT": "hello",
        "MAX_NEW_TOKENS": "3",
        "TEMPERATURE": "0.5",
        "PAUSE_SECONDS": "0",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    callbacks = []
    # Avoid leaving callbacks that outlive monkeypatch's CUDA replacements.
    monkeypatch.setattr(atexit, "register", callbacks.append)
    return callbacks


@pytest.fixture
def inference(script_environment, monkeypatch):
    for name, value in {
        "is_available": True,
        "is_initialized": True,
        "get_device_name": "Test GPU",
        "get_device_properties": SimpleNamespace(total_memory=8 * 1024**3),
        "memory_allocated": 1024**3,
        "memory_reserved": 2 * 1024**3,
        "max_memory_allocated": 3 * 1024**3,
        "reset_peak_memory_stats": None,
    }.items():
        monkeypatch.setattr(torch.cuda, name, Mock(return_value=value))

    model = Mock()
    model.to.return_value = model
    model.parameters.return_value = [
        SimpleNamespace(device=torch.device("cuda:0"), numel=lambda: 12)
    ]
    model.buffers.return_value = []
    inputs = {"input_ids": torch.tensor([[1, 2]])}
    tokenizer = Mock(eos_token_id=0)
    tokenizer.return_value.to.return_value = inputs
    tokenizer.decode.return_value = "hello world"

    def generate(**kwargs):
        assert torch.is_inference_mode_enabled()
        return torch.tensor([[1, 2, 3]])

    model.generate.side_effect = generate
    model_loader = Mock(return_value=model)
    tokenizer_loader = Mock(return_value=tokenizer)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", model_loader)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", tokenizer_loader)
    return SimpleNamespace(
        model=model,
        tokenizer=tokenizer,
        model_loader=model_loader,
        tokenizer_loader=tokenizer_loader,
        inputs=inputs,
        callbacks=script_environment,
    )


def run_script():
    return runpy.run_module("src.gpu_inference", run_name="__main__")


def test_generation_uses_configured_prompt_and_gpu(inference, capsys):
    run_script()
    inference.model_loader.assert_called_once_with(
        "test-local-model", dtype=torch.float16, token=None
    )
    inference.tokenizer_loader.assert_called_once_with("test-local-model", token=None)
    inference.model.to.assert_called_once_with(torch.device("cuda:0"))
    inference.model.eval.assert_called_once()
    inference.tokenizer.assert_called_once_with("hello", return_tensors="pt")
    inference.tokenizer.return_value.to.assert_called_once_with(torch.device("cuda:0"))
    inference.model.generate.assert_called_once_with(
        **inference.inputs,
        max_new_tokens=3,
        do_sample=True,
        temperature=0.5,
        pad_token_id=0,
    )
    output = capsys.readouterr().out
    assert "hello world" in output
    assert "RESULT: SUCCESS" in output


def test_no_cuda_fails_before_loading(inference, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        run_script()
    inference.model_loader.assert_not_called()
    inference.tokenizer_loader.assert_not_called()


@pytest.mark.parametrize("location", ["parameters", "buffers"])
def test_cpu_offloading_is_rejected(inference, location, capsys):
    getattr(inference.model, location).return_value = [
        SimpleNamespace(device=torch.device("cpu"), numel=lambda: 12)
    ]
    with pytest.raises(RuntimeError, match="cuda:0"):
        run_script()
    inference.model.generate.assert_not_called()
    assert "RESULT: SUCCESS" not in capsys.readouterr().out


@pytest.mark.parametrize("failure_stage", ["load", "generate"])
def test_failure_still_registers_memory_report(inference, failure_stage, capsys):
    target = (
        inference.model_loader if failure_stage == "load" else inference.model.generate
    )
    target.side_effect = torch.OutOfMemoryError("test allocation failure")
    with pytest.raises(torch.OutOfMemoryError, match="test allocation failure"):
        run_script()
    assert "RESULT: SUCCESS" not in capsys.readouterr().out
    assert len(inference.callbacks) == 1
    inference.callbacks[0]()
    report = capsys.readouterr().out
    assert f"Final allocated bytes: {1024**3}" in report
    assert f"Peak allocated bytes: {3 * 1024**3}" in report


def test_memory_report_without_cuda_initialization(inference, monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        run_script()
    capsys.readouterr()
    inference.callbacks[0]()
    report = capsys.readouterr().out
    assert "Finished (UTC):" in report
    assert "allocated bytes" not in report
    torch.cuda.memory_allocated.assert_not_called()


def test_optional_inspection_pause(inference, monkeypatch):
    monkeypatch.setenv("PAUSE_SECONDS", "2")
    sleep = Mock()
    monkeypatch.setattr(time, "sleep", sleep)
    run_script()
    sleep.assert_called_once_with(2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU unavailable")
def test_real_cuda_generation_with_local_tiny_model(
    script_environment, tmp_path, monkeypatch, capsys
):
    """Exercise real loading, fp16 transfer, and generation without Hub access."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    backend = Tokenizer(
        WordLevel({"[UNK]": 0, "[EOS]": 1, "hello": 2}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]"
    )
    tokenizer.save_pretrained(tmp_path)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=3,
                n_positions=16,
                n_embd=16,
                n_layer=1,
                n_head=1,
                bos_token_id=1,
                eos_token_id=1,
                pad_token_id=1,
            )
        )
    model.save_pretrained(tmp_path)
    monkeypatch.setenv("MODEL_ID", str(tmp_path))
    with torch.random.fork_rng(devices=[0]):
        result = run_script()
    assert result["output"].device == torch.device("cuda:0")
    assert 1 < result["output"].shape[1] <= 4
    assert {p.dtype for p in result["model"].parameters()} == {torch.float16}
    assert "RESULT: SUCCESS" in capsys.readouterr().out
