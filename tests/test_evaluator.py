"""Unit tests for the offline evaluation setup."""

from unittest.mock import MagicMock

import pytest

from src import evaluator as evaluator_module


@pytest.fixture
def expected_results():
    """Provide the result shape returned by Hugging Face's QA evaluator."""
    return {
        "exact_match": 42.0,
        "f1": 61.5,
        "total_time_in_seconds": 1.2,
        "samples_per_second": 10.0,
        "latency_in_seconds": 0.1,
    }


@pytest.fixture
def mocked_dependencies(monkeypatch, expected_results):
    """Replace network/model-dependent collaborators with local mocks."""
    data = object()
    load_dataset = MagicMock(return_value=data)
    task_evaluator = MagicMock()
    task_evaluator.compute.return_value = expected_results
    evaluator_factory = MagicMock(return_value=task_evaluator)

    monkeypatch.setattr(evaluator_module, "load_dataset", load_dataset)
    monkeypatch.setattr(evaluator_module, "evaluator", evaluator_factory)

    return data, load_dataset, evaluator_factory, task_evaluator


def test_run_evaluation_uses_validation_by_default(
    mocked_dependencies, expected_results
):
    """The default call loads AdversarialQA validation and returns QA metrics."""
    data, load_dataset, evaluator_factory, task_evaluator = mocked_dependencies

    results = evaluator_module.run_evaluation(model_name="any-compatible-qa-model")

    assert results == expected_results
    assert results["exact_match"] == 42.0
    assert results["f1"] == 61.5
    load_dataset.assert_called_once_with(
        "UCLNLP/adversarial_qa", "adversarialQA", split="validation"
    )
    evaluator_factory.assert_called_once_with("question-answering")
    task_evaluator.compute.assert_called_once_with(
        model_or_pipeline="any-compatible-qa-model",
        data=data,
        metric="squad",
    )


@pytest.mark.parametrize("split", ["train", "validation"])
def test_run_evaluation_forwards_supported_splits(mocked_dependencies, split):
    """Callers can choose either labeled AdversarialQA split."""
    _, load_dataset, _, _ = mocked_dependencies

    evaluator_module.run_evaluation(model_name="another-qa-model", split=split)

    assert load_dataset.call_args.kwargs["split"] == split
