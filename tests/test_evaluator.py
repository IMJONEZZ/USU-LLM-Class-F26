"""
Tests for the evaluator in src/evaluator.py.

We mock the BERT model, tokenizer, and dataset loading rather than
downloading real ones during tests. This keeps tests fast, deterministic,
and independent of network access -- important for CI, where downloading a
full BERT model and dataset on every test run would be slow and fragile.
"""

import torch

from src.evaluator import (
    compute_metrics,
    load_eval_dataset,
    load_model_and_tokenizer,
    predict_labels,
    run_evaluation,
)

# ---------------------------------------------------------------------------
# FAKE MODEL / TOKENIZER
# Minimal stand-ins that mimic just enough of the real Hugging Face
# interface for predict_labels() to work correctly against them.
# ---------------------------------------------------------------------------


class FakeOutputs:
    """Mimics a Hugging Face model output object with a `.logits` attribute."""

    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    """
    A fake model that always predicts a fixed label for every input, so we
    can precisely control and verify predict_labels()'s behavior.
    """

    def __init__(self, fixed_label=1):
        self.fixed_label = fixed_label
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def __call__(self, **inputs):
        batch_size = inputs["input_ids"].shape[0]
        # Build logits so that argmax always picks self.fixed_label
        logits = torch.zeros((batch_size, 2))
        logits[:, self.fixed_label] = 10.0
        return FakeOutputs(logits)


class FakeTokenizer:
    """A fake tokenizer that returns dummy tensors, ignoring the actual text content."""

    def __call__(self, texts, padding=True, truncation=True, return_tensors="pt"):
        batch_size = len(texts)
        return {
            "input_ids": torch.ones((batch_size, 4), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 4), dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# compute_metrics() TESTS
# ---------------------------------------------------------------------------


def test_compute_metrics_perfect_predictions():
    """When predictions exactly match references, accuracy and F1 should both be 1.0."""
    predictions = [1, 0, 1, 0, 1]
    references = [1, 0, 1, 0, 1]
    metrics = compute_metrics(predictions, references)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0


def test_compute_metrics_all_wrong():
    """When every prediction is wrong, accuracy should be 0.0."""
    predictions = [0, 1, 0, 1]
    references = [1, 0, 1, 0]
    metrics = compute_metrics(predictions, references)
    assert metrics["accuracy"] == 0.0


def test_compute_metrics_partial_correct():
    """Accuracy should reflect the exact fraction of correct predictions."""
    predictions = [1, 1, 0, 0]
    references = [1, 0, 0, 0]
    metrics = compute_metrics(predictions, references)
    assert metrics["accuracy"] == 0.75


def test_compute_metrics_returns_expected_keys():
    """The returned dict should contain both 'accuracy' and 'f1' keys."""
    metrics = compute_metrics([1, 0], [1, 0])
    assert "accuracy" in metrics
    assert "f1" in metrics


# ---------------------------------------------------------------------------
# predict_labels() TESTS
# ---------------------------------------------------------------------------


def test_predict_labels_returns_correct_number_of_predictions():
    """predict_labels should return exactly one prediction per input text."""
    model = FakeModel(fixed_label=1)
    tokenizer = FakeTokenizer()
    texts = ["good movie", "bad movie", "great film"]

    predictions = predict_labels(model, tokenizer, texts, batch_size=2)

    assert len(predictions) == len(texts)


def test_predict_labels_returns_expected_label():
    """A model that always predicts label 1 should produce all 1s."""
    model = FakeModel(fixed_label=1)
    tokenizer = FakeTokenizer()
    texts = ["a", "b", "c"]

    predictions = predict_labels(model, tokenizer, texts)

    assert all(p == 1 for p in predictions)


def test_predict_labels_handles_batching_correctly():
    """
    Predictions should be correct and in order even when the number of texts
    isn't evenly divisible by batch_size, exercising the batching loop's
    boundary handling.
    """
    model = FakeModel(fixed_label=0)
    tokenizer = FakeTokenizer()
    texts = ["a", "b", "c", "d", "e"]  # 5 texts, batch_size=2 -> uneven last batch

    predictions = predict_labels(model, tokenizer, texts, batch_size=2)

    assert len(predictions) == 5
    assert all(p == 0 for p in predictions)


def test_predict_labels_empty_input():
    """predict_labels should return an empty list when given no texts."""
    model = FakeModel()
    tokenizer = FakeTokenizer()

    predictions = predict_labels(model, tokenizer, [])

    assert predictions == []


# ---------------------------------------------------------------------------
# load_model_and_tokenizer() TESTS
# Mocked so we don't download a real model from the Hub during tests.
# ---------------------------------------------------------------------------


def test_load_model_and_tokenizer_calls_eval(monkeypatch):
    """The loaded model should have .eval() called on it, to disable dropout etc."""
    fake_model = FakeModel()

    def fake_model_from_pretrained(name):
        return fake_model

    def fake_tokenizer_from_pretrained(name):
        return FakeTokenizer()

    monkeypatch.setattr(
        "src.evaluator.AutoModelForSequenceClassification.from_pretrained",
        fake_model_from_pretrained,
    )
    monkeypatch.setattr(
        "src.evaluator.AutoTokenizer.from_pretrained",
        fake_tokenizer_from_pretrained,
    )

    model, tokenizer = load_model_and_tokenizer("fake-model-name")

    assert model.eval_called is True
    assert isinstance(tokenizer, FakeTokenizer)


# ---------------------------------------------------------------------------
# load_eval_dataset() TESTS
# Mocked so we don't download the real SST-2 dataset during tests.
# ---------------------------------------------------------------------------


def test_load_eval_dataset_limits_num_samples(monkeypatch):
    """When num_samples is given, the returned dataset should be limited to that size."""

    class FakeDataset:
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def select(self, indices):
            return FakeDataset([self.data[i] for i in indices])

    fake_dataset = FakeDataset(list(range(1000)))

    def fake_load_dataset(name, config, split):
        return fake_dataset

    monkeypatch.setattr("src.evaluator.load_dataset", fake_load_dataset)

    result = load_eval_dataset(num_samples=50)

    assert len(result) == 50


def test_load_eval_dataset_no_limit_returns_full_dataset(monkeypatch):
    """When num_samples is None, the full dataset should be returned unmodified."""

    class FakeDataset:
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

    fake_dataset = FakeDataset(list(range(10)))

    def fake_load_dataset(name, config, split):
        return fake_dataset

    monkeypatch.setattr("src.evaluator.load_dataset", fake_load_dataset)

    result = load_eval_dataset(num_samples=None)

    assert len(result) == 10


# ---------------------------------------------------------------------------
# run_evaluation() TEST
# Full pipeline test, with every external dependency mocked.
# ---------------------------------------------------------------------------


def test_run_evaluation_full_pipeline(monkeypatch):
    """
    The full evaluation pipeline should load a model/tokenizer/dataset, run
    predictions, and return a metrics dict -- with every external call
    mocked so this test doesn't hit the network.
    """

    class FakeDataset(dict):
        def select(self, indices):
            return self

    fake_dataset = FakeDataset(
        {"sentence": ["good", "bad", "great", "awful"], "label": [1, 0, 1, 0]}
    )

    monkeypatch.setattr(
        "src.evaluator.load_model_and_tokenizer",
        lambda: (FakeModel(fixed_label=1), FakeTokenizer()),
    )
    monkeypatch.setattr(
        "src.evaluator.load_eval_dataset",
        lambda num_samples: fake_dataset,
    )

    metrics = run_evaluation(num_samples=4)

    assert "accuracy" in metrics
    assert "f1" in metrics
    # FakeModel always predicts 1, and half the labels are 1 -> accuracy 0.5
    assert metrics["accuracy"] == 0.5
