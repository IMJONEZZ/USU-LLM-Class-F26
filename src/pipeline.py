from zenml import pipeline, step


@step
def data_prep() -> list[float]:
    return [1.0, 2.0, 3.0]


@step
def train_model(data: list[float]) -> float:
    return sum(data) / len(data)


@pipeline
def training_pipeline() -> None:
    data = data_prep()
    train_model(data)


if __name__ == "__main__":
    training_pipeline()
