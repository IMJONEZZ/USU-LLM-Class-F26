"""Unsloth loader contract; CPU CI does not install or initialize CUDA kernels."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import peft
import pytest
import torch
import transformers
from fastapi.testclient import TestClient

from src import server


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": server.MODEL_ID})
    )
    (tmp_path / "adapter_model.safetensors").touch()
    monkeypatch.setenv("ADAPTER_PATH", str(tmp_path))
    monkeypatch.delenv("DEVICE", raising=False)
    return tmp_path


def test_missing_weights(adapter):
    (adapter / "adapter_model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="adapter_model.safetensors"):
        server.load_runtime()


def test_wrong_base(adapter):
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "other/model"})
    )
    with pytest.raises(ValueError, match="must be trained on"):
        server.load_runtime()


@pytest.mark.parametrize("device", ["cpu", "invalid"])
def test_reject_cpu_and_unknown_device(adapter, monkeypatch, device):
    monkeypatch.setenv("DEVICE", device)
    with pytest.raises(ValueError, match="make serve-transformers"):
        server.load_runtime()


def test_missing_cuda(adapter, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        server.load_runtime()


@pytest.mark.parametrize("pad_token_id", [None, 0])
def test_unsloth_loads_pinned_base_and_saved_adapter(
    adapter, monkeypatch, pad_token_id
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    fast_model = Mock()
    base, model = Mock(), Mock()
    fast_model.from_pretrained.return_value = (base, Mock())
    fast_model.patch_peft_model.return_value = model
    monkeypatch.setitem(
        sys.modules, "unsloth", SimpleNamespace(FastLanguageModel=fast_model)
    )
    peft_loader = Mock(return_value=model)
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", peft_loader)
    tokenizer = SimpleNamespace(pad_token_id=pad_token_id, eos_token="eos")
    tokenizer_loader = Mock(return_value=tokenizer)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_loader)

    with TestClient(server.create_app(server.load_runtime)) as client:
        health = client.get("/health").json()
        assert health["backend"] == "unsloth"
        assert health["device"] == "cuda"
        assert health["adapter"] == str(adapter)
        assert health["revision"] == server.MODEL_REVISION

    fast_model.from_pretrained.assert_called_once_with(
        model_name=server.MODEL_ID,
        revision=server.MODEL_REVISION,
        max_seq_length=1280,
        dtype=torch.float16,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        device_map={"": 0},
        use_exact_model_name=True,
        use_gradient_checkpointing=False,
    )
    peft_loader.assert_called_once_with(base, adapter, is_trainable=False)
    fast_model.patch_peft_model.assert_called_once_with(
        model, use_gradient_checkpointing=False
    )
    fast_model.for_inference.assert_called_once_with(model)
    model.requires_grad_.assert_called_once_with(False)
    tokenizer_loader.assert_called_once_with(adapter, local_files_only=True)
    if pad_token_id is None:
        assert tokenizer.pad_token == "eos"
