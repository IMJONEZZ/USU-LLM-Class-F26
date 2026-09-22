import datasets
import evaluate
import transformers


def test_imports():
    assert datasets and evaluate and transformers
