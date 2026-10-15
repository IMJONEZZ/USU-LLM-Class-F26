"""A three step ZenML pipeline, adapted from their MLOps starter example.

Loads the sklearn digits dataset, trains a random forest on it, and scores it
on a held out split. The point isn't the model, it's having a pipeline with
several connected steps and real artifacts moving between them so there's
something to look at in the ZenML dashboard.

ZenML needs Python below 3.12 because it imports distutils, and this repo pins
3.14, so it runs out of its own virtualenv:

    .zenml-venv/bin/python pipelines/digits_pipeline.py
"""

from typing import Annotated

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from zenml import pipeline, step

# Annotating the return types is what gives the artifacts readable names in the
# dashboard. Without it they show up as output_0, output_1 and so on.


@step
def load_data(
    test_size: float = 0.25, seed: int = 42
) -> tuple[
    Annotated[np.ndarray, "X_train"],
    Annotated[np.ndarray, "X_test"],
    Annotated[np.ndarray, "y_train"],
    Annotated[np.ndarray, "y_test"],
]:
    """Load the digits dataset and split it."""
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=test_size, random_state=seed
    )
    print(f"train {X_train.shape}, test {X_test.shape}")
    return X_train, X_test, y_train, y_test


@step
def train_model(
    X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 100, seed: int = 42
) -> Annotated[ClassifierMixin, "model"]:
    """Fit a random forest on the training split."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    model.fit(X_train, y_train)
    print(f"trained on {len(X_train)} rows with {n_estimators} trees")
    return model


@step
def evaluate_model(
    model: ClassifierMixin, X_test: np.ndarray, y_test: np.ndarray
) -> Annotated[float, "accuracy"]:
    """Score the fitted model on the held out split."""
    accuracy = float(model.score(X_test, y_test))
    print(f"accuracy: {accuracy:.4f}")
    return accuracy


@pipeline
def digits_pipeline(test_size: float = 0.25, n_estimators: int = 100):
    """Wire the three steps together.

    Nothing here runs the steps directly. Calling them records the dependencies
    between them, and ZenML works out the order from what each one needs.
    """
    X_train, X_test, y_train, y_test = load_data(test_size=test_size)
    model = train_model(X_train, y_train, n_estimators=n_estimators)
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    run = digits_pipeline()
    print(f"\nfinished: {run.name}")
