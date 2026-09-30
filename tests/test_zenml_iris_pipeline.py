"""Real Iris step tests and graph preparation, without a ZenML server or store."""

from unittest.mock import Mock

import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.utils.validation import check_is_fitted

from src import zenml_iris_pipeline as iris


def test_split_is_repeatable_aligned_and_has_no_leakage():
    first = iris.load_data.entrypoint()
    second = iris.load_data.entrypoint()
    X_train, X_test, y_train, y_test = first
    assert X_train.shape == (120, 4)
    assert X_test.shape == (30, 4)
    assert X_train.index.equals(y_train.index)
    assert X_test.index.equals(y_test.index)
    assert set(X_train.index).isdisjoint(X_test.index)
    source = load_iris(as_frame=True)
    pd.testing.assert_frame_equal(
        pd.concat([X_train, X_test]).sort_index(), source.data
    )
    pd.testing.assert_series_equal(
        pd.concat([y_train, y_test]).sort_index(), source.target
    )
    for original, repeated in zip(first, second, strict=True):
        if isinstance(original, pd.DataFrame):
            pd.testing.assert_frame_equal(original, repeated)
        else:
            pd.testing.assert_series_equal(original, repeated)


def test_real_training_and_evaluation(capsys):
    X_train, X_test, y_train, y_test = iris.load_data.entrypoint()
    model = iris.train_model.entrypoint(X_train, y_train)
    check_is_fitted(model)
    assert set(model.classes_) == {0, 1, 2}
    accuracy = iris.evaluate_model.entrypoint(model, X_test, y_test)
    expected = (model.predict(X_test.to_numpy()) == y_test.to_numpy()).mean()
    assert isinstance(accuracy, float)
    assert accuracy == pytest.approx(expected)
    # This checks correctness; the assignment does not specify a quality target.
    assert 0.0 <= accuracy <= 1.0
    assert f"Test accuracy: {accuracy:.4f}" in capsys.readouterr().out


def test_evaluation_reports_known_fraction_correct(capsys):
    model = Mock()
    model.predict.return_value = [0, 0, 2, 1]
    X_test = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y_test = pd.Series([0, 1, 2, 2])
    accuracy = iris.evaluate_model.entrypoint(model, X_test, y_test)
    assert accuracy == 0.5
    assert "Test accuracy: 0.5000" in capsys.readouterr().out


def test_pipeline_connects_loading_training_and_evaluation():
    # prepare builds ZenML's graph; it does not submit or execute a pipeline run.
    pipeline = iris.iris_training_pipeline.copy()
    pipeline.prepare()
    invocations = pipeline.invocations
    assert set(invocations) == {"load_data", "train_model", "evaluate_model"}
    assert invocations["load_data"].upstream_steps == set()
    assert invocations["train_model"].upstream_steps == {"load_data"}
    assert invocations["evaluate_model"].upstream_steps == {"load_data", "train_model"}
