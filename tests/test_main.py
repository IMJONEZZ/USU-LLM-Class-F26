from unittest.mock import patch

from src.main import compare_models


@patch("src.main.evaluate_model")
def test_compare_models(mock_evaluate_model):
    mock_evaluate_model.side_effect = [{"accuracy": 0.9}, {"accuracy": 0.8}]

    result = compare_models("model-a", "model-b", "fake-dataset")

    assert result == [
        {"name": "model-a", "score": 0.9},
        {"name": "model-b", "score": 0.8},
    ]
