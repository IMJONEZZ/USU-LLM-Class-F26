import math
import sys
from types import ModuleType, SimpleNamespace

import nltk.translate.bleu_score as bleu
import pandas as pd
import pytest
import torch

from src import evaluator
from src.evaluator import Evaluator, data, generate_prediction


class MockModel:
    """A deterministic stand-in for a text-generation model."""

    def __init__(self, predictions: dict[str, str]) -> None:
        self.predictions = predictions
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.predictions[prompt]


@pytest.fixture
def expected_results() -> pd.Series:
    """Load a small set of reference outputs into the format Evaluator uses."""
    return pd.Series(data["expected"].iloc[:2].tolist())


@pytest.fixture
def predicted_results() -> pd.Series:
    """Load the corresponding model outputs into the format Evaluator uses."""
    return pd.Series(data["predicted"].iloc[:2].tolist())


def test_initialization_stores_evaluation_algorithm() -> None:
    evaluator = Evaluator(bleu.sentence_bleu)

    assert evaluator.eval_alg is bleu.sentence_bleu


def test_expected_and_predicted_results_load_as_series(
    expected_results: pd.Series, predicted_results: pd.Series
) -> None:
    assert expected_results.tolist() == data["expected"].iloc[:2].tolist()
    assert predicted_results.tolist() == data["predicted"].iloc[:2].tolist()


def test_split_sentence_returns_words_and_separates_pipe_delimited_steps() -> None:
    evaluator = Evaluator(bleu.sentence_bleu)

    result = evaluator._Evaluator__split_sentence("**LOCK** the door.|**LEAVE** now.")

    assert result == ["**LOCK**", "the", "door.", "**LEAVE**", "now."]


def test_predict_uses_fake_model_for_each_input() -> None:
    inputs = pd.Series(["first instruction", "second instruction"])
    model = MockModel(
        {
            "first instruction": "first prediction",
            "second instruction": "second prediction",
        }
    )
    evaluator = Evaluator(bleu.sentence_bleu)

    predictions = evaluator.predict(inputs, model)

    assert predictions.tolist() == ["first prediction", "second prediction"]
    assert model.calls == inputs.tolist()


def test_evaluate_applies_brevity_penalty_for_missing_words() -> None:
    evaluator = Evaluator(
        lambda references, hypothesis: bleu.sentence_bleu(
            references, hypothesis, weights=(0.5, 0.5)
        )
    )

    scores = evaluator.evaluate(
        pd.Series(["one two three four"]), pd.Series(["one two"])
    )

    # Both predicted words are correct, but the short hypothesis receives BLEU's
    # brevity penalty: exp(1 - reference_length / hypothesis_length).
    assert scores == pytest.approx([math.exp(1 - 4 / 2)])


def test_evaluate_rejects_mismatched_result_lengths() -> None:
    evaluator = Evaluator(bleu.sentence_bleu)

    with pytest.raises(ValueError, match="Length of actual values"):
        evaluator.evaluate(pd.Series(["expected"]), pd.Series(["first", "second"]))


class FakeBatch(dict):
    def to(self, _device: str) -> FakeBatch:
        return self


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self) -> None:
        self.prompt = ""

    def __call__(self, prompt: str, *, return_tensors: str) -> FakeBatch:
        assert return_tensors == "pt"
        self.prompt = prompt
        return FakeBatch(input_ids=torch.tensor([[10, 11]]))

    def decode(self, token_ids: torch.Tensor, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        assert token_ids.tolist() == [12, 13]
        return " generated result "


class FakeGenerationModel:
    def __init__(self) -> None:
        self.generation_kwargs = {}

    def generate(self, **kwargs) -> torch.Tensor:
        self.generation_kwargs = kwargs
        return torch.tensor([[10, 11, 12, 13]])


def test_generate_prediction_uses_training_prompt_and_decodes_only_new_tokens() -> None:
    tokenizer = FakeTokenizer()
    model = FakeGenerationModel()

    prediction = generate_prediction(model, tokenizer, "lock the valve", device="cpu")

    assert tokenizer.prompt == "Convert to PPA:\nlock the valve\nOutput:\n"
    assert prediction == "generated result"
    assert model.generation_kwargs["max_new_tokens"] == 512
    assert model.generation_kwargs["do_sample"] is False


def test_remote_evaluation_pipeline_with_mocked_model_downloads(
    monkeypatch, capsys
) -> None:
    calls = {}
    test_data = pd.DataFrame(
        {
            "input": ["turn the valve", "release the latch"],
            "expected": ["**TURN** the valve.", "**RELEASE** the latch."],
            "predicted": ["unused", "unused"],
        }
    )

    class FakeDownloadTokenizer:
        pad_token = None
        eos_token = "<eos>"

    class FakeBaseModel:
        def to(self, device):
            calls["device"] = device
            return self

    class FakeAdapter:
        def eval(self):
            calls["eval_called"] = True

    class FakeHfApi:
        def __init__(self, token):
            calls["hf_token"] = token

        def whoami(self):
            return {"name": "test-user"}

    def fake_module(name, **attrs):
        module = ModuleType(name)
        for attr, value in attrs.items():
            setattr(module, attr, value)
        monkeypatch.setitem(sys.modules, name, module)

    fake_tokenizer = FakeDownloadTokenizer()
    fake_base_model = FakeBaseModel()
    fake_adapter = FakeAdapter()
    fake_module("huggingface_hub", HfApi=FakeHfApi)
    fake_module(
        "peft",
        PeftModel=SimpleNamespace(
            from_pretrained=lambda base, repo, **kwargs: (
                calls.update(adapter_repo=repo, adapter_token=kwargs["token"])
                or fake_adapter
            )
        ),
    )
    fake_module(
        "transformers",
        AutoModelForCausalLM=SimpleNamespace(
            from_pretrained=lambda model_id, **kwargs: (
                calls.update(base_model_id=model_id, load_token=kwargs["token"])
                or fake_base_model
            )
        ),
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda model_id, **kwargs: (
                calls.update(
                    tokenizer_model_id=model_id, tokenizer_token=kwargs["token"]
                )
                or fake_tokenizer
            )
        ),
    )
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(evaluator, "data", test_data)
    monkeypatch.setattr(evaluator, "modal", None)
    monkeypatch.setattr(
        evaluator,
        "generate_prediction",
        lambda model, tokenizer, source: f"prediction for {source}",
    )

    result = evaluator._evaluate_model()

    assert fake_tokenizer.pad_token == "<eos>"
    assert calls["base_model_id"] == "meta-llama/Llama-3.2-1B"
    assert calls["tokenizer_model_id"] == "meta-llama/Llama-3.2-1B"
    assert calls["adapter_repo"] == "test-user/ppa-llama-3.2-1b-lora"
    assert calls["device"] == "cuda"
    assert calls["eval_called"] is True
    assert result["predictions"] == [
        "prediction for turn the valve",
        "prediction for release the latch",
    ]
    assert len(result["scores"]) == 2
    assert result["average_bleu"] == pytest.approx(sum(result["scores"]) / 2)
    assert "Average BLEU:" in capsys.readouterr().out
