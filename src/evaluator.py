"""Evaluate binary sentiment classifiers on SST-2.

The :class:`Evaluator` depends only on an injected prediction function.  Model
and dataset loading live behind optional, lazy imports so unit tests can stay
offline while the real BERT evaluation uses the exact pinned artifacts below.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

NEGATIVE_LABEL = 0
POSITIVE_LABEL = 1
LABEL_NAMES = {NEGATIVE_LABEL: "negative", POSITIVE_LABEL: "positive"}

MODEL_ID = "gokuls/bert-base-uncased-sst2"
MODEL_REVISION = "853d9058744ab305b05870e5bcf1e41438b66485"
DATASET_ID = "nyu-mll/glue"
DATASET_CONFIG = "sst2"
DATASET_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
EXPECTED_VALIDATION_SIZE = 872

Predictor = Callable[[Sequence[str]], Sequence[int]]


@dataclass(frozen=True)
class ClassMetrics:
    """Metrics for one sentiment class."""

    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class EvaluationResult:
    """Complete, serializable result of a binary sentiment evaluation."""

    example_count: int
    accuracy: float
    macro_f1: float
    per_class: dict[str, ClassMetrics]
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]
    majority_class_baseline: float
    accuracy_confidence_interval_95: tuple[float, float]
    accuracy_threshold_met: bool
    confidence_above_baseline: bool
    quality_bar_met: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert nested dataclasses and tuples to JSON-compatible values."""

        return asdict(self)


def _validate_binary_labels(values: Sequence[int], name: str) -> tuple[int, ...]:
    """Return labels as integers after checking the binary label contract."""

    converted = tuple(int(value) for value in values)
    invalid = sorted(set(converted) - set(LABEL_NAMES))
    if invalid:
        raise ValueError(f"{name} must contain only 0 and 1; found {invalid}")
    return converted


def _safe_divide(numerator: int, denominator: int) -> float:
    """Use zero for an undefined class metric."""

    return numerator / denominator if denominator else 0.0


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    """Linearly interpolate a percentile from an already sorted sample."""

    position = (len(sorted_values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - fraction)
        + sorted_values[upper_index] * fraction
    )


def bootstrap_accuracy_interval(
    labels: Sequence[int],
    predictions: Sequence[int],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Calculate a seeded percentile bootstrap interval for accuracy."""

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not labels:
        raise ValueError("labels cannot be empty")
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length")

    correct = [
        expected == predicted for expected, predicted in zip(labels, predictions)
    ]
    rng = random.Random(seed)
    bootstrap_accuracies = sorted(
        sum(rng.choices(correct, k=len(correct))) / len(correct)
        for _ in range(n_resamples)
    )

    tail_probability = (1.0 - confidence) / 2.0
    return (
        _percentile(bootstrap_accuracies, tail_probability),
        _percentile(bootstrap_accuracies, 1.0 - tail_probability),
    )


def calculate_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 42,
    accuracy_threshold: float = 0.90,
) -> EvaluationResult:
    """Calculate deterministic binary-classification metrics and quality checks."""

    labels = _validate_binary_labels(labels, "labels")
    predictions = _validate_binary_labels(predictions, "predictions")
    if not labels:
        raise ValueError("labels cannot be empty")
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length")

    confusion = [[0, 0], [0, 0]]
    for expected, predicted in zip(labels, predictions, strict=True):
        confusion[expected][predicted] += 1

    per_class: dict[str, ClassMetrics] = {}
    class_f1_scores = []
    for label, label_name in LABEL_NAMES.items():
        true_positive = confusion[label][label]
        predicted_count = confusion[0][label] + confusion[1][label]
        support = sum(confusion[label])
        precision = _safe_divide(true_positive, predicted_count)
        recall = _safe_divide(true_positive, support)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        per_class[label_name] = ClassMetrics(precision, recall, f1, support)
        class_f1_scores.append(f1)

    example_count = len(labels)
    accuracy = (confusion[0][0] + confusion[1][1]) / example_count
    macro_f1 = sum(class_f1_scores) / len(class_f1_scores)
    majority_class_baseline = max(map(sum, confusion)) / example_count
    confidence_interval = bootstrap_accuracy_interval(
        labels,
        predictions,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )
    accuracy_threshold_met = accuracy >= accuracy_threshold
    confidence_above_baseline = confidence_interval[0] > majority_class_baseline

    return EvaluationResult(
        example_count=example_count,
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class=per_class,
        confusion_matrix=(
            (confusion[0][0], confusion[0][1]),
            (confusion[1][0], confusion[1][1]),
        ),
        majority_class_baseline=majority_class_baseline,
        accuracy_confidence_interval_95=confidence_interval,
        accuracy_threshold_met=accuracy_threshold_met,
        confidence_above_baseline=confidence_above_baseline,
        quality_bar_met=accuracy_threshold_met and confidence_above_baseline,
    )


class Evaluator:
    """Evaluate text with an injected predictor and reproducible bootstrap settings."""

    def __init__(
        self,
        predictor: Predictor,
        *,
        confidence: float = 0.95,
        n_resamples: int = 10_000,
        seed: int = 42,
        accuracy_threshold: float = 0.90,
    ) -> None:
        self.predictor = predictor
        self.confidence = confidence
        self.n_resamples = n_resamples
        self.seed = seed
        self.accuracy_threshold = accuracy_threshold

    def evaluate(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
    ) -> EvaluationResult:
        """Predict labels for text and calculate all configured metrics."""

        texts = tuple(texts)
        labels = tuple(labels)
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        predictions = tuple(self.predictor(texts))
        return calculate_metrics(
            labels,
            predictions,
            confidence=self.confidence,
            n_resamples=self.n_resamples,
            seed=self.seed,
            accuracy_threshold=self.accuracy_threshold,
        )


def validate_id2label(id2label: Mapping[int | str, str]) -> None:
    """Fail fast unless a checkpoint explicitly declares SST-2 label semantics."""

    normalized = {int(index): label.lower() for index, label in id2label.items()}
    if normalized != LABEL_NAMES:
        raise ValueError(
            f"Expected id2label {LABEL_NAMES}, but checkpoint declares {normalized}"
        )


class BertSST2Predictor:
    """Batch predictor for the exact pinned BERT SST-2 checkpoint."""

    def __init__(self, *, batch_size: int = 32, device: str | None = None) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        import torch
        from transformers import (
            AutoConfig,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        validate_id2label(config.id2label)
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
        )
        self.device = device or self._best_device(torch)
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.torch = torch

    @staticmethod
    def _best_device(torch_module: Any) -> str:
        if torch_module.cuda.is_available():
            return "cuda"
        if torch_module.backends.mps.is_available():
            return "mps"
        return "cpu"

    def __call__(self, texts: Sequence[str]) -> list[int]:
        predictions: list[int] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
            with self.torch.inference_mode():
                logits = self.model(**encoded).logits
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
        return predictions


def load_sst2_validation() -> tuple[list[str], list[int]]:
    """Load the complete, pinned 872-example SST-2 validation split."""

    from datasets import load_dataset

    dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split="validation",
        revision=DATASET_REVISION,
    )
    if len(dataset) != EXPECTED_VALIDATION_SIZE:
        raise RuntimeError(
            f"Expected {EXPECTED_VALIDATION_SIZE} validation examples, got {len(dataset)}"
        )
    return list(dataset["sentence"]), [int(label) for label in dataset["label"]]


def run_bert_evaluation(*, batch_size: int = 32) -> EvaluationResult:
    """Sanity-check label semantics, then evaluate pinned BERT on pinned SST-2."""

    predictor = BertSST2Predictor(batch_size=batch_size)
    sanity_texts = [
        "A wonderful movie that I loved.",
        "A terrible movie that I hated.",
    ]
    sanity_predictions = predictor(sanity_texts)
    if sanity_predictions != [POSITIVE_LABEL, NEGATIVE_LABEL]:
        raise RuntimeError(
            f"Sentiment sanity check failed: expected [1, 0], got {sanity_predictions}"
        )

    texts, labels = load_sst2_validation()
    return Evaluator(predictor).evaluate(texts, labels)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the manual network-backed evaluation and print reproducible JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    result = run_bert_evaluation(batch_size=args.batch_size)
    report = {
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "dataset": {
            "id": DATASET_ID,
            "config": DATASET_CONFIG,
            "revision": DATASET_REVISION,
            "split": "validation",
        },
        "bootstrap": {"confidence": 0.95, "resamples": 10_000, "seed": 42},
        "result": result.to_dict(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":  # pragma: no cover - manual network-backed entry point
    main()
