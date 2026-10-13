import sys
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest
import torch

from src import evaluator
from src.evaluator import Evaluator, generate_prediction


def test_predict_calls_model_once_per_question(capsys):
    model = lambda text: f"answer: {text}"
    predictions = Evaluator(lambda *_: 1).predict(pd.Series(["first", "second"]), model)

    assert predictions.tolist() == ["answer: first", "answer: second"]
    assert "2 / 2 predictions completed." in capsys.readouterr().out


def test_predict_empty_series_returns_empty_series():
    assert Evaluator(lambda *_: 1).predict(pd.Series(dtype=str), str).empty


def test_evaluate_splits_steps_and_applies_evaluation_function():
    observed = []

    def score(references, hypothesis):
        observed.append((references, hypothesis))
        return len(hypothesis)

    scores = Evaluator(score).evaluate(
        pd.Series(["one two|three"]), pd.Series(["one|two three"])
    )

    assert scores == [3]
    assert observed == [([["one", "two", "three"]], ["one", "two", "three"])]


def test_evaluate_rejects_different_lengths():
    with pytest.raises(ValueError, match="Length of actual values"):
        Evaluator(lambda *_: 0).evaluate(pd.Series(["one"]), pd.Series([]))


class FakeBatch(dict):
    def to(self, device):
        self.device = device
        return self


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, prompt, *, return_tensors):
        self.prompt = prompt
        assert return_tensors == "pt"
        self.batch = FakeBatch(input_ids=torch.tensor([[10, 11]]))
        return self.batch

    def decode(self, token_ids, *, skip_special_tokens):
        assert token_ids.tolist() == [12, 13]
        assert skip_special_tokens
        return " prediction "


class FakeGenerationModel:
    def generate(self, **kwargs):
        self.kwargs = kwargs
        return torch.tensor([[10, 11, 12, 13]])


def test_generate_prediction_formats_prompt_and_decodes_new_tokens():
    tokenizer = FakeTokenizer()
    model = FakeGenerationModel()

    result = generate_prediction(
        model, tokenizer, "how?", device="cpu", max_new_tokens=17
    )

    assert tokenizer.prompt == "Question:\nhow?\nAnswer:\n"
    assert tokenizer.batch.device == "cpu"
    assert result == "prediction"
    assert model.kwargs["max_new_tokens"] == 17
    assert model.kwargs["do_sample"] is False
    assert model.kwargs["pad_token_id"] == 0
    assert model.kwargs["eos_token_id"] == 2


def test_evaluate_model_with_mocked_downloads(monkeypatch, capsys):
    calls = {}

    class Tokenizer:
        pad_token = None
        eos_token = "eos"

    class BaseModel:
        def to(self, device):
            calls["device"] = device
            return self

    class Adapter:
        def eval(self):
            calls["eval"] = True

    class HfApi:
        def __init__(self, token):
            calls["token"] = token

        def whoami(self):
            return {"name": "student"}

    def install_module(name, **attrs):
        module = ModuleType(name)
        vars(module).update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    tokenizer = Tokenizer()
    base_model = BaseModel()
    adapter = Adapter()
    install_module("huggingface_hub", HfApi=HfApi)
    install_module(
        "peft",
        PeftModel=SimpleNamespace(
            from_pretrained=lambda base, repo, **kwargs: (
                calls.update(repo=repo, adapter_token=kwargs["token"]) or adapter
            )
        ),
    )
    install_module(
        "transformers",
        AutoModelForCausalLM=SimpleNamespace(
            from_pretrained=lambda model_id, **kwargs: (
                calls.update(base_id=model_id, model_token=kwargs["token"])
                or base_model
            )
        ),
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda model_id, **kwargs: (
                calls.update(tokenizer_id=model_id) or tokenizer
            )
        ),
    )
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setattr(
        evaluator,
        "data",
        pd.DataFrame({"questions": ["q1", "q2"], "answers": ["a1", "a2"]}),
    )
    monkeypatch.setattr(
        evaluator, "generate_prediction", lambda _model, _tokenizer, q: f"pred {q}"
    )

    result = evaluator._evaluate_model()

    assert tokenizer.pad_token == "eos"
    assert calls == {
        "token": "secret",
        "base_id": "meta-llama/Llama-3.2-1B",
        "model_token": "secret",
        "tokenizer_id": "meta-llama/Llama-3.2-1B",
        "repo": "student/A6-llama-3.2-1b-lora",
        "adapter_token": "secret",
        "device": "cuda",
        "eval": True,
    }
    assert result["question"].tolist() == ["q1", "q2"]
    assert result["prediction"] == ["pred q1", "pred q2"]
    assert result["answer"].tolist() == ["a1", "a2"]
    assert "2/2: generated prediction" in capsys.readouterr().out


def test_modal_app_registers_remote_and_local_entrypoint(monkeypatch, capsys):
    calls = {}

    class App:
        def function(self, **kwargs):
            calls["function"] = kwargs
            return lambda fn: fn

        def local_entrypoint(self):
            return lambda fn: fn

    fake_modal = SimpleNamespace(
        App=lambda name: calls.update(app=name) or App(),
        Secret=SimpleNamespace(
            from_local_environ=lambda names: calls.update(secret=names) or "secret"
        ),
        Image=SimpleNamespace(
            debian_slim=lambda: SimpleNamespace(
                pip_install=lambda *args: calls.update(packages=args) or "image"
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setenv("HF_TOKEN", "secret")
    reloaded = __import__("importlib").reload(evaluator)
    monkeypatch.setattr(
        reloaded,
        "evaluate_model",
        SimpleNamespace(remote=lambda: {"prediction": ["answer"]}),
    )

    reloaded.main()

    assert calls["app"] == "ppa-model-evaluation"
    assert calls["function"] == {
        "image": "image",
        "gpu": "T4",
        "secrets": ["secret"],
    }
    assert "answer" in capsys.readouterr().out
