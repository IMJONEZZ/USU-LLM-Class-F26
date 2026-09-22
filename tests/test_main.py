from unittest.mock import patch

from src.dataloader import load_imdb_subset
from src.main import compare_models

MODEL_A = "lvwerra/distilbert-imdb"
MODEL_B = "distilbert-base-uncased-finetuned-sst-2-english"


@patch("src.main.evaluate_model")
def test_compare_models(mock_evaluate_model):
    mock_evaluate_model.side_effect = [{"accuracy": 0.9}, {"accuracy": 0.8}]

    result = compare_models("model-a", "model-b", "fake-dataset")

    assert result == [
        {"name": "model-a", "score": 0.9},
        {"name": "model-b", "score": 0.8},
    ]


if __name__ == "__main__":
    dataset = load_imdb_subset()
    results = compare_models(MODEL_A, MODEL_B, dataset)
    print(results)
