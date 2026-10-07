"""Reports refuse mismatched runs and never fill missing experiments with numbers."""

import json
from copy import deepcopy

import pytest

from src import training_report as report


@pytest.fixture
def reports():
    meta = {
        "signature": "data",
        "model": "llama",
        "revision": "commit",
        "split": "test",
        "batch_size": 1,
        "device": "cuda",
        "max_new_tokens": 32,
        "generation": "greedy",
        "normalization": "whitespace",
        "empty_treatment": "zero",
        "adapter": None,
        "training_run": None,
    }
    examples = [
        {
            "example_id": "a",
            "prompt": "context",
            "reference": "target",
            "prediction": "wrong",
            "exact": False,
        },
        {
            "example_id": "b",
            "prompt": "other",
            "reference": "other target",
            "prediction": "wrong",
            "exact": False,
        },
    ]
    before = {
        "status": "complete",
        "metadata": meta,
        "examples": examples,
        "metrics": {
            "exact_span_match": 0.0,
            "rouge": {"rouge1": 0.0, "rougeL": 0.0},
            "bleu": {"bleu": 0.0},
            "bertscore": {"f1": 0.0},
        },
    }
    training = {
        "metadata": {"signature": "data"},
        "training_seconds": 2.0,
        "timing_scope": "loops",
        "peak_allocated_bytes": 100,
        "peak_reserved_bytes": 200,
        "initial_dev_loss": 3.0,
        "best_epoch": 1,
        "best_dev_loss": 2.0,
        "history": [{"epoch": 1, "train_loss": 1.0, "dev_loss": 2.0}],
    }
    after = deepcopy(before)
    after["metadata"]["adapter"] = "best"
    after["metadata"]["training_run"] = {
        **training["metadata"],
        "checkpoint_epoch": 1,
        "checkpoint_dev_loss": 2.0,
    }
    after["examples"][0].update(prediction="target", exact=True)
    after["metrics"]["exact_span_match"] = 0.5
    return before, after, training


def test_report_and_cli(reports, tmp_path):
    draft = report.render_report(*reports)
    assert "improved" in draft and "2.000 seconds" in draft
    assert '"after": "target"' in draft and "(failure)" in draft
    paths = []
    for i, data in enumerate(reports):
        path = tmp_path / f"{i}.json"
        path.write_text(json.dumps(data))
        paths.append(str(path))
    output = tmp_path / "report.md"
    report.main(
        [
            "--before",
            paths[0],
            "--after",
            paths[1],
            "--training",
            paths[2],
            "--output",
            str(output),
        ]
    )
    assert output.read_text() == draft


@pytest.mark.parametrize(
    "mutation",
    [
        "incomplete",
        "settings",
        "baseline_adapter",
        "training",
        "checkpoint",
        "examples",
    ],
)
def test_report_rejects_mismatches(reports, mutation):
    before, after, training = reports
    if mutation == "incomplete":
        after["status"] = "generated_unscored"
    elif mutation == "settings":
        after["metadata"]["max_new_tokens"] = 5
    elif mutation == "baseline_adapter":
        before["metadata"]["adapter"] = "not-baseline"
    elif mutation == "training":
        training["metadata"]["signature"] = "different"
    elif mutation == "checkpoint":
        after["metadata"]["training_run"]["checkpoint_epoch"] = 2
    else:
        after["examples"][0]["reference"] = "different"
    with pytest.raises(ValueError):
        report.render_report(*reports)


def test_report_no_success_does_not_invent_examples(reports):
    reports[1]["examples"][0]["exact"] = False
    reports[1]["metrics"]["exact_span_match"] = 0.0
    draft = report.render_report(*reports)
    assert "No successes" in draft and "stayed the same" in draft
