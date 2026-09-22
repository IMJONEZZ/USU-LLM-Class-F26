import evaluate

LABEL_MAPPING = {"NEGATIVE": 0, "POSITIVE": 1}


def evaluate_model(model_name: str, dataset) -> dict:
    """Evaluate a text-classification model's accuracy on the given dataset."""
    return evaluate.evaluator("text-classification").compute(
        model_or_pipeline=model_name,
        data=dataset,
        metric="accuracy",
        label_mapping=LABEL_MAPPING,
    )
