"""Offline API checks, including loading a real tiny Llama + LoRA checkpoint."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import peft
import pytest
import torch
import transformers
from fastapi.testclient import TestClient
from peft import LoraConfig, PeftModel, get_peft_model
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    BatchEncoding,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

from src.transformers_app import server


@pytest.fixture
def adapter(tmp_path):
    path = tmp_path / "adapter"
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "base_model_name_or_path": server.MODEL_ID,
                "r": 4,
                "lora_alpha": 8,
                "target_modules": ["q_proj", "v_proj"],
            }
        )
    )
    (path / "adapter_model.safetensors").touch()
    return path


@pytest.fixture
def runtime():
    tokenizer = Mock(pad_token_id=0)
    tokenizer.apply_chat_template.return_value = "formatted chat"
    tokenizer.return_value = BatchEncoding(
        {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2)}
    )
    tokenizer.decode.return_value = "  three  "
    model = Mock()
    model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
    return server.Runtime(model, tokenizer, "cpu", Path("final-adapter"))


def test_api_lifecycle_and_generation(runtime):
    loader = Mock(return_value=runtime)
    app = server.create_app(loader)
    loader.assert_not_called()
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health == {
            "status": "ready",
            "model": server.MODEL_ID,
            "revision": server.MODEL_REVISION,
            "adapter": "final-adapter",
            "device": "cpu",
            "backend": "transformers-peft",
        }
        for _ in range(2):
            response = client.post("/generate", json={"prompt": "  question  "})
            assert response.status_code == 200
            result = response.json()
            assert result["text"] == "three"
            assert result["prompt_tokens"] == result["generated_tokens"] == 2
            assert result["elapsed_seconds"] >= 0
        loader.assert_called_once_with()
        runtime.tokenizer.apply_chat_template.assert_called_with(
            [{"role": "user", "content": "question"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        runtime.tokenizer.assert_called_with(
            "formatted chat", add_special_tokens=False, return_tensors="pt"
        )
        assert runtime.model.generate.call_args.kwargs["do_sample"] is False
        assert runtime.model.generate.call_args.kwargs["max_new_tokens"] == 128
        assert runtime.tokenizer.decode.call_args.args[0].tolist() == [3, 4]
    assert not hasattr(app.state, "runtime")


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"prompt": " "},
        {"prompt": "x" * 8001},
        {"prompt": 123},
        {"prompt": "q", "max_new_tokens": 0},
        {"prompt": "q", "max_new_tokens": 257},
        {"prompt": "q", "max_new_tokens": True},
        {"prompt": "q", "max_new_tokens": 1.5},
        {"prompt": "q", "unknown": "option"},
    ],
)
def test_invalid_requests_do_not_generate(runtime, payload):
    with TestClient(server.create_app(lambda: runtime)) as client:
        assert client.post("/generate", json=payload).status_code == 422
    runtime.model.generate.assert_not_called()


def test_token_limit_and_recovery(runtime):
    with TestClient(server.create_app(lambda: runtime)) as client:
        runtime.tokenizer.return_value = BatchEncoding(
            {"input_ids": torch.ones(1, server.MAX_INPUT_TOKENS + 1, dtype=torch.long)}
        )
        response = client.post("/generate", json={"prompt": "long input"})
        assert response.status_code == 422
        assert "after chat formatting" in response.json()["detail"]
        runtime.model.generate.assert_not_called()
        runtime.tokenizer.return_value = BatchEncoding(
            {"input_ids": torch.tensor([[1, 2]])}
        )
        assert client.post("/generate", json={"prompt": "short"}).status_code == 200


def test_missing_adapter(monkeypatch, tmp_path):
    monkeypatch.setenv("ADAPTER_PATH", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="ADAPTER_PATH"):
        server.load_runtime()


def test_wrong_base(monkeypatch, adapter):
    monkeypatch.setenv("ADAPTER_PATH", str(adapter))
    config_path = adapter / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["base_model_name_or_path"] = "wrong/base"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="must be trained on"):
        server.load_runtime()


@pytest.mark.parametrize("device", ["invalid", "cuda"])
def test_invalid_device(monkeypatch, adapter, device):
    monkeypatch.setenv("ADAPTER_PATH", str(adapter))
    monkeypatch.setenv("DEVICE", device)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises((ValueError, RuntimeError), match="DEVICE"):
        server.load_runtime()


def test_cuda_loader_configuration(monkeypatch, adapter):
    monkeypatch.setenv("ADAPTER_PATH", str(adapter))
    monkeypatch.delenv("DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    tokenizer = SimpleNamespace(pad_token_id=None, eos_token="eos")
    tokenizer_loader = Mock(return_value=tokenizer)
    base_loader = Mock()
    adapter_loader = Mock()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", base_loader
    )
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", adapter_loader)
    runtime = server.load_runtime()
    assert runtime.device == "cuda"
    assert tokenizer.pad_token == "eos"
    base_loader.assert_called_once_with(
        server.MODEL_ID,
        revision=server.MODEL_REVISION,
        dtype=torch.float16,
        attn_implementation="sdpa",
    )
    adapter_loader.assert_called_once_with(
        base_loader.return_value, adapter, is_trainable=False
    )
    runtime.model.to.assert_called_once_with("cuda")
    runtime.model.eval.assert_called_once_with()


def test_real_lora_checkpoint_through_api(monkeypatch, tmp_path):
    """Exercise PEFT attachment and generation without downloading the 1B model."""
    base_path, adapter_path = tmp_path / "base", tmp_path / "adapter"
    base = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=8,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=64,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    base.save_pretrained(base_path)
    model = get_peft_model(
        base,
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.125)
    model.peft_config["default"].base_model_name_or_path = server.MODEL_ID
    model.save_pretrained(adapter_path, save_embedding_layers=False)
    backend = Tokenizer(
        models.WordLevel(
            {
                "[PAD]": 0,
                "[BOS]": 1,
                "[EOS]": 2,
                "[UNK]": 3,
                "user": 4,
                "assistant": 5,
                "hello": 6,
                "answer": 7,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        chat_template="{{ bos_token }} user {{ messages[0]['content'] }} assistant",
    )
    tokenizer.save_pretrained(adapter_path)
    original_loader = transformers.AutoModelForCausalLM.from_pretrained
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: original_loader(base_path, **kwargs),
    )
    monkeypatch.setenv("ADAPTER_PATH", str(adapter_path))
    monkeypatch.setenv("DEVICE", "auto")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    app = server.create_app()
    with TestClient(app) as client:
        loaded = app.state.runtime.model
        assert isinstance(loaded, PeftModel)
        assert not any(p.requires_grad for p in loaded.parameters())
        assert not loaded.training
        for name, parameter in loaded.named_parameters():
            if "lora_B" in name:
                assert torch.all(parameter == 0.125)
        response = client.post(
            "/generate", json={"prompt": "hello", "max_new_tokens": 2}
        )
        assert response.status_code == 200
        assert response.json()["prompt_tokens"] == 4
        assert 1 <= response.json()["generated_tokens"] <= 2
        assert client.get("/health").json()["device"] == "cpu"


def test_startup_failure_is_not_ready():
    app = server.create_app(Mock(side_effect=RuntimeError("load failed")))
    with pytest.raises(RuntimeError, match="load failed"), TestClient(app):
        pass
