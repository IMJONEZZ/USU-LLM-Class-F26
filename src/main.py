from src.dataloader import load_imdb_subset
from src.evaluator import evaluate_model

MODEL_A = "lvwerra/distilbert-imdb"
MODEL_B = "distilbert-base-uncased-finetuned-sst-2-english"


def compare_models(model_a_name: str, model_b_name: str, dataset) -> list[dict]:
    """Evaluate two models on the same dataset and return their accuracy scores."""
    return [
        {
            "name": model_a_name,
            "score": evaluate_model(model_a_name, dataset)["accuracy"],
        },
        {
            "name": model_b_name,
            "score": evaluate_model(model_b_name, dataset)["accuracy"],
        },
    ]


if __name__ == "__main__":
    dataset = load_imdb_subset()
    results = compare_models(MODEL_A, MODEL_B, dataset)
    print(results)
