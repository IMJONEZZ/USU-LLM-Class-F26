from typing import Annotated

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from zenml import pipeline, step


@step
def load_data() -> tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """Load and split the Iris dataset."""
    iris = load_iris(as_frame=True)

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    return X_train, X_test, y_train, y_test


@step
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Annotated[ClassifierMixin, "trained_model"]:
    """Train an SVC classifier."""
    model = SVC(gamma=0.001)

    model.fit(
        X_train.to_numpy(),
        y_train.to_numpy(),
    )

    return model


@step
def evaluate_model(
    model: ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Annotated[float, "test_accuracy"]:
    """Evaluate the trained model."""
    predictions = model.predict(X_test.to_numpy())

    accuracy = accuracy_score(
        y_test.to_numpy(),
        predictions,
    )

    print(f"Test accuracy: {accuracy:.4f}")

    return float(accuracy)


@pipeline
def iris_training_pipeline():
    """Train and evaluate an SVC on Iris."""
    X_train, X_test, y_train, y_test = load_data()

    model = train_model(
        X_train=X_train,
        y_train=y_train,
    )

    evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
    )


if __name__ == "__main__":
    iris_training_pipeline()
