from unittest.mock import MagicMock, patch

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
