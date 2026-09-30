from src.pipeline import data_prep, train_model


def test_data_prep_returns_placeholder_data():
    result = data_prep.entrypoint()

    assert result == [1.0, 2.0, 3.0]


def test_train_model_returns_mean_of_data():
    result = train_model.entrypoint([2.0, 4.0, 6.0])

    assert result == 4.0
