"""Tests for the MRPC evaluator.

Nothing in here downloads anything. The model is a fake classifier with answers
I picked, the dataset is a handful of rows built in the test, and the metric is
sklearn instead of the GLUE one from evaluate. I checked those two agree before
swapping: on the same inputs both give accuracy 0.75 and f1 0.8, which makes
sense because the GLUE metric is a thin wrapper around sklearn anyway.
"""

import pytest
from sklearn.metrics import accuracy_score, f1_score

from src.evaluator import (
    DEFAULT_CONFIG,
    LABEL_MAPPING,
    build_predictor,
    evaluate_predictor,
    load_glue_metric,
    load_mrpc,
    main,
    majority_baseline,
)


class LocalMetric:
    """Stands in for the GLUE metric so the tests stay offline."""

    def __init__(self):
        self.calls = []

    def compute(self, predictions, references):
        self.calls.append((list(predictions), list(references)))
        return {
            "accuracy": accuracy_score(references, predictions),
            "f1": f1_score(references, predictions, zero_division=0),
        }


class FakeClassifier:
    """Pretends to be the transformers pipeline and returns canned labels."""

    def __init__(self, labels):
        self.labels = labels
        self.seen_inputs = None
        self.seen_batch_size = None

    def __call__(self, inputs, batch_size=None):
        self.seen_inputs = inputs
        self.seen_batch_size = batch_size
        return [{"label": label} for label in self.labels]


class FakeRows:
    """The bits of a datasets.Dataset that load_mrpc actually touches."""

    def __init__(self, first, second, labels):
        self.columns = {"sentence1": first, "sentence2": second, "label": labels}

    def __getitem__(self, name):
        return self.columns[name]

    def __len__(self):
        return len(self.columns["label"])

    def select(self, indexes):
        keep = list(indexes)
        return FakeRows(*([self.columns[c][i] for i in keep] for c in self.columns))


def pairs_for(count):
    return [(f"first {i}", f"second {i}") for i in range(count)]


# --- the baseline, which is the whole reason this evaluator exists ----------


def test_baseline_of_nothing_is_zero():
    assert majority_baseline([]) == 0.0


def test_baseline_of_one_class_is_everything():
    assert majority_baseline([1, 1, 1, 1]) == 1.0


def test_baseline_matches_the_real_mrpc_split():
    """279 equivalent and 129 not, which is the actual validation split."""
    labels = [1] * 279 + [0] * 129
    assert round(majority_baseline(labels), 4) == 0.6838


def test_baseline_picks_the_bigger_class_not_the_first_one():
    assert majority_baseline([0, 1, 1]) == pytest.approx(2 / 3)


# --- scoring ----------------------------------------------------------------


def test_scores_match_numbers_worked_out_by_hand():
    # 3 of 4 right, and of the two it called equivalent one was correct, so
    # precision 1/2 and recall 1/1 which puts f1 at 0.8.
    labels = [1, 0, 0, 1]
    results = evaluate_predictor(
        lambda pairs: [1, 1, 0, 1], pairs_for(4), labels, metric=LocalMetric()
    )
    assert results["accuracy"] == 0.75
    assert results["f1"] == pytest.approx(0.8)


def test_lift_is_accuracy_minus_baseline():
    labels = [1, 1, 1, 0]
    results = evaluate_predictor(
        lambda pairs: [1, 1, 1, 1], pairs_for(4), labels, metric=LocalMetric()
    )
    assert results["majority_baseline"] == 0.75
    assert results["accuracy"] == 0.75
    # A model that just guesses the common class buys nothing, and this is the
    # number that says so.
    assert results["lift_over_baseline"] == 0.0


def test_a_perfect_predictor_scores_one():
    labels = [1, 0, 1, 0]
    results = evaluate_predictor(
        lambda pairs: list(labels), pairs_for(4), labels, metric=LocalMetric()
    )
    assert results["accuracy"] == 1.0
    assert results["lift_over_baseline"] == 0.5


def test_results_carry_the_count_and_the_predictions():
    results = evaluate_predictor(
        lambda pairs: [1, 0], pairs_for(2), [1, 0], metric=LocalMetric()
    )
    assert results["n"] == 2
    assert results["predictions"] == [1, 0]


def test_the_injected_metric_is_the_one_that_gets_used():
    metric = LocalMetric()
    evaluate_predictor(lambda pairs: [1, 0], pairs_for(2), [1, 0], metric=metric)
    assert metric.calls == [([1, 0], [1, 0])]


def test_more_labels_than_pairs_is_rejected():
    with pytest.raises(ValueError, match="sentence pairs"):
        evaluate_predictor(lambda pairs: [], pairs_for(2), [1, 0, 1])


def test_a_predictor_that_loses_rows_is_rejected():
    # This one matters. Without the check the metric would happily score a
    # short list against the full labels and give a wrong number quietly.
    with pytest.raises(ValueError, match="predictions"):
        evaluate_predictor(
            lambda pairs: [1], pairs_for(3), [1, 0, 1], metric=LocalMetric()
        )


# --- the model wrapper, with the pipeline faked out -------------------------


def test_labels_come_back_as_zeros_and_ones():
    fake = FakeClassifier(["LABEL_1", "LABEL_0", "LABEL_1"])
    predict = build_predictor(classifier=fake)
    assert predict(pairs_for(3)) == [1, 0, 1]


def test_the_mapping_i_checked_is_the_one_in_the_file():
    """If this ever flips, accuracy drops to about 12% and looks like a bad model."""
    assert LABEL_MAPPING == {"LABEL_0": 0, "LABEL_1": 1}


def test_a_custom_mapping_is_honored():
    fake = FakeClassifier(["LABEL_0", "LABEL_1"])
    predict = build_predictor(
        label_mapping={"LABEL_0": 1, "LABEL_1": 0}, classifier=fake
    )
    assert predict(pairs_for(2)) == [1, 0]


def test_an_unknown_label_blows_up():
    fake = FakeClassifier(["SOMETHING_ELSE"])
    with pytest.raises(KeyError):
        build_predictor(classifier=fake)(pairs_for(1))


def test_no_pairs_means_the_model_is_never_called():
    fake = FakeClassifier([])
    assert build_predictor(classifier=fake)([]) == []
    assert fake.seen_inputs is None


def test_sentences_are_sent_as_a_pair_not_glued_together():
    # BERT needs these as two separate segments. If they got concatenated the
    # model would still return something, it would just be wrong.
    fake = FakeClassifier(["LABEL_1"])
    build_predictor(batch_size=4, classifier=fake)([("left side", "right side")])
    assert fake.seen_inputs == [{"text": "left side", "text_pair": "right side"}]
    assert fake.seen_batch_size == 4


# --- loading the data and the metric ----------------------------------------


@pytest.fixture
def fake_dataset(monkeypatch):
    rows = FakeRows(
        ["a1", "b1", "c1"],
        ["a2", "b2", "c2"],
        [1, 0, 1],
    )
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: rows)
    return rows


def test_load_mrpc_pairs_the_two_sentence_columns(fake_dataset):
    pairs, labels = load_mrpc()
    assert pairs == [("a1", "a2"), ("b1", "b2"), ("c1", "c2")]
    assert labels == [1, 0, 1]


def test_limit_takes_the_first_rows(fake_dataset):
    pairs, labels = load_mrpc(limit=2)
    assert pairs == [("a1", "a2"), ("b1", "b2")]
    assert labels == [1, 0]


def test_a_limit_bigger_than_the_split_is_fine(fake_dataset):
    pairs, _ = load_mrpc(limit=500)
    assert len(pairs) == 3


def test_the_metric_asked_for_is_glue_mrpc(monkeypatch):
    asked = {}

    def fake_load(name, config):
        asked["name"] = name
        asked["config"] = config
        return "metric"

    monkeypatch.setattr("evaluate.load", fake_load)
    assert load_glue_metric() == "metric"
    assert asked == {"name": "glue", "config": DEFAULT_CONFIG}


# --- the script end to end --------------------------------------------------


def test_main_prints_the_numbers(monkeypatch, capsys):
    monkeypatch.setattr(
        "src.evaluator.load_mrpc", lambda limit=None: (pairs_for(4), [1, 1, 1, 0])
    )
    monkeypatch.setattr("src.evaluator.build_predictor", lambda: lambda p: [1, 1, 1, 0])
    monkeypatch.setattr("src.evaluator.load_glue_metric", LocalMetric)

    results = main()
    printed = capsys.readouterr().out

    assert results["accuracy"] == 1.0
    assert "examples           4" in printed
    assert "accuracy           1.0000" in printed
    assert "majority baseline  0.7500" in printed
    assert "lift over baseline +0.2500" in printed
