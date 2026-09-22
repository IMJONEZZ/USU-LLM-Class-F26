from unittest.mock import MagicMock, patch

import pytest

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
