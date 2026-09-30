from src.pipeline import svc_trainer, training_data_loader


def test_training_data_loader_returns_iris_train_test_split():
    X_train, X_test, y_train, y_test = training_data_loader.entrypoint()

    assert len(X_train) + len(X_test) == 150
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_svc_trainer_returns_fitted_model_and_accuracy():
    X_train, X_test, y_train, _y_test = training_data_loader.entrypoint()

    model, train_acc = svc_trainer.entrypoint(X_train, y_train)

    assert 0.0 <= train_acc <= 1.0
    assert model.predict(X_test.to_numpy()) is not None
