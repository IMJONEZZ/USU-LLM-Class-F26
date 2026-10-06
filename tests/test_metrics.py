from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config import EMBEDDING_MODEL_ID
from src.metrics import (
    is_valid_category,
    load_embedding_model,
    normalize_label,
    semantic_similarity,
)

CATEGORIES = ["HR", "INFORMATION-TECHNOLOGY", "BPO"]


def make_model(vectors):
    model = MagicMock()
    model.encode.return_value = np.array(vectors, dtype=float)
    return model


@patch("src.metrics.SentenceTransformer")
def test_load_embedding_model_uses_configured_model_id(mock_sentence_transformer):
    model = load_embedding_model()

    mock_sentence_transformer.assert_called_once_with(EMBEDDING_MODEL_ID)
    assert model is mock_sentence_transformer.return_value


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("  HR\n", "hr"),
        ("INFORMATION-TECHNOLOGY", "information technology"),
        ("Information   Technology.", "information technology"),
        ("", ""),
    ],
)
def test_normalize_label(raw, expected):
    assert normalize_label(raw) == expected


@pytest.mark.parametrize(
    "generated",
    ["HR", " hr ", "Hr.", "information technology", "INFORMATION-TECHNOLOGY"],
)
def test_is_valid_category_accepts_formatting_variants(generated):
    assert is_valid_category(generated, CATEGORIES)


@pytest.mark.parametrize(
    "generated", ["", "   ", "Human Resources", "HR ADMINISTRATOR", "banana"]
)
def test_is_valid_category_rejects_non_categories(generated):
    assert not is_valid_category(generated, CATEGORIES)


def test_semantic_similarity_is_one_for_identical_embeddings():
    model = make_model([[1.0, 0.0], [1.0, 0.0]])
    assert semantic_similarity("a", "b", model) == pytest.approx(1.0)


def test_semantic_similarity_is_zero_for_orthogonal_embeddings():
    model = make_model([[1.0, 0.0], [0.0, 1.0]])
    assert semantic_similarity("a", "b", model) == pytest.approx(0.0)


def test_semantic_similarity_is_hand_checked_cosine():
    model = make_model([[1.0, 1.0], [1.0, 0.0]])
    assert semantic_similarity("a", "b", model) == pytest.approx(2**-0.5)


def test_semantic_similarity_encodes_generated_then_expected_once():
    model = make_model([[1.0, 0.0], [1.0, 0.0]])
    semantic_similarity("generated", "expected", model)
    model.encode.assert_called_once_with(["generated", "expected"])


@pytest.mark.slow
def test_semantic_similarity_real_model_ranks_near_miss_above_unrelated():
    model = load_embedding_model()

    assert semantic_similarity("HR", "HR", model) == pytest.approx(1.0, abs=1e-3)
    near_miss = semantic_similarity("Health care", "HEALTHCARE", model)
    unrelated = semantic_similarity("Aviation", "HEALTHCARE", model)
    assert near_miss > unrelated
