import math

import nltk.translate.bleu_score as bleu
import pandas as pd
import pytest

from src.evaluator import Evaluator, data


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
