from src.evaluator import evaluate_model


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
