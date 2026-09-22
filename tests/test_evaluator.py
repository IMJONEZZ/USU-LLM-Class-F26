from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from src.dataloader import load_imdb_subset
from src.evaluator import evaluate_model


@patch("src.evaluator.evaluate.evaluator")
def test_evaluate_model_passes_correct_arguments(mock_evaluator_factory):
    mock_task_evaluator = MagicMock()
    mock_task_evaluator.compute.return_value = {"accuracy": 0.9}
    mock_evaluator_factory.return_value = mock_task_evaluator

    fake_dataset = object()
    result = evaluate_model("distilbert-base-uncased", fake_dataset)

    mock_evaluator_factory.assert_called_once_with("text-classification")
    mock_task_evaluator.compute.assert_called_once_with(
        model_or_pipeline="distilbert-base-uncased",
        data=fake_dataset,
        metric="accuracy",
        label_mapping={"NEGATIVE": 0, "POSITIVE": 1},
    )
    assert result == {"accuracy": 0.9}


@pytest.mark.slow
def test_evaluate_model_real_inference():
    dataset = load_imdb_subset(n=10)
    result = evaluate_model("lvwerra/distilbert-imdb", dataset)

    assert "accuracy" in result
    assert 0 <= result["accuracy"] <= 1


def test_label_postive_direction_is_correct():
    obviously_positive = Dataset.from_dict(
        {
            "text": [
                "This was the best movie I have ever seen. Absolutely brilliant, a true masterpiece."
            ],
            "label": [1],
        }
    )

    result = evaluate_model("lvwerra/distilbert-imdb", obviously_positive)

    assert result["accuracy"] == 1.0


def test_label_negative_direction_is_correct():
    obviously_positive = Dataset.from_dict(
        {
            "text": ["This was the worst movie I have ever seen. Abhorrent."],
            "label": [0],
        }
    )

    result = evaluate_model("lvwerra/distilbert-imdb", obviously_positive)

    assert result["accuracy"] == 1.0


@pytest.mark.slow
def test_matches_published_benchmark_accuracy():
    dataset = load_imdb_subset(n=1000)

    result = evaluate_model("lvwerra/distilbert-imdb", dataset)

    assert 0.85 <= result["accuracy"] <= 1.0


def test_end_to_end_reproducibility():
    dataset_first = load_imdb_subset(n=10, seed=42)
    dataset_second = load_imdb_subset(n=10, seed=42)

    first_result = evaluate_model("lvwerra/distilbert-imdb", dataset_first)
    second_result = evaluate_model("lvwerra/distilbert-imdb", dataset_second)

    assert first_result["accuracy"] == second_result["accuracy"]
