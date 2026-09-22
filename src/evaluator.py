"""
Evaluate a pretrained BERT model on the SST-2 sentiment classification task
using Hugging Face's `evaluate` library.

Model: textattack/bert-base-uncased-SST-2
    A BERT model fine-tuned specifically for binary sentiment classification
    on the SST-2 dataset (part of the GLUE benchmark).

Dataset: glue/sst2 (validation split)
    Short sentences labeled 0 (negative) or 1 (positive). We use the
    validation split since SST-2's test split has no public labels.
"""

import evaluate
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "textattack/bert-base-uncased-SST-2"
DATASET_NAME = "nyu-mll/glue"
DATASET_CONFIG = "sst2"
DATASET_SPLIT = "validation"


def load_model_and_tokenizer(model_name=MODEL_NAME):
    """Load the pretrained BERT model and its matching tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def load_eval_dataset(
    dataset_name=DATASET_NAME,
    dataset_config=DATASET_CONFIG,
    split=DATASET_SPLIT,
    num_samples=None,
):
    """
    Load the SST-2 validation dataset. Optionally limit to `num_samples`
    examples, which keeps evaluation fast during development and testing.
    """
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    return dataset


def predict_labels(model, tokenizer, texts, batch_size=16):
    """
    Run the model on a list of texts and return predicted labels as a list
    of ints (0 = negative, 1 = positive), matching SST-2's label convention.
    """
    predictions = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        batch_preds = torch.argmax(outputs.logits, dim=-1).tolist()
        predictions.extend(batch_preds)

    return predictions


def compute_metrics(predictions, references):
    """
    Compute accuracy and F1 score comparing model predictions against the
    true SST-2 labels.
    """
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    accuracy_result = accuracy_metric.compute(
        predictions=predictions, references=references
    )
    f1_result = f1_metric.compute(predictions=predictions, references=references)

    return {
        "accuracy": accuracy_result["accuracy"],
        "f1": f1_result["f1"],
    }


def run_evaluation(num_samples=200):
    """
    Full evaluation pipeline: load the model, load a subset of SST-2, run
    predictions, and compute accuracy/F1 against the true labels.
    """
    model, tokenizer = load_model_and_tokenizer()
    dataset = load_eval_dataset(num_samples=num_samples)

    texts = dataset["sentence"]
    references = dataset["label"]

    predictions = predict_labels(model, tokenizer, texts)
    metrics = compute_metrics(predictions, references)

    return metrics


if __name__ == "__main__":  # pragma: no cover
    results = run_evaluation(num_samples=200)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
