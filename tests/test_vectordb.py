"""Tests for document embedding and cosine-similarity retrieval."""

from typing import ClassVar

import numpy as np
import pytest

from src import vectordb


class FakeEmbeddingModel:
    """Return small, predictable vectors without downloading a model."""

    vectors: ClassVar[dict[str, list[float]]] = {
        "same phrase": [1.0, 0.0],
        "perpendicular phrase": [0.0, 1.0],
        "diagonal phrase": [1.0, 1.0],
    }

    def encode_document(self, texts, convert_to_numpy):
        assert convert_to_numpy is True
        return np.array([self.vectors[text] for text in texts], dtype=np.float32)

    def encode_query(self, query, convert_to_numpy):
        assert convert_to_numpy is True
        return np.array(self.vectors[query], dtype=np.float32)


@pytest.fixture
def populated_collection(tmp_path, monkeypatch):
    """Create a real local Chroma collection with deterministic embeddings."""
    monkeypatch.setattr(vectordb, "DATABASE_PATH", tmp_path)
    model = FakeEmbeddingModel()
    texts = list(model.vectors)
    collection = vectordb.create_collection(model, texts)
    return model, collection


def test_query_returns_document_with_highest_cosine_similarity(
    populated_collection,
):
    model, collection = populated_collection

    result = vectordb.query_collection(model, collection, "same phrase", n_results=1)

    assert result[0]["id"] == "document-0"
    assert result[0]["document"] == "same phrase"
    assert result[0]["cosine_similarity"] == pytest.approx(1.0)


def test_query_calculates_cosine_similarity_for_each_result(
    populated_collection,
):
    model, collection = populated_collection

    results = vectordb.query_collection(model, collection, "same phrase", n_results=3)
    similarities = {
        result["document"]: result["cosine_similarity"] for result in results
    }

    assert similarities["same phrase"] == pytest.approx(1.0)
    assert similarities["perpendicular phrase"] == pytest.approx(0.0)
    assert similarities["diagonal phrase"] == pytest.approx(1 / np.sqrt(2))


def test_create_collection_rejects_no_documents():
    with pytest.raises(ValueError, match="at least one non-empty document"):
        vectordb.create_collection(FakeEmbeddingModel(), [])


def test_query_collection_rejects_empty_query(populated_collection):
    model, collection = populated_collection

    with pytest.raises(ValueError, match="query must not be empty"):
        vectordb.query_collection(model, collection, "   ")


def test_main_prints_query_results(monkeypatch, capsys):
    model = FakeEmbeddingModel()
    collection = type("Collection", (), {"count": lambda self: 3})()
    monkeypatch.setattr(vectordb, "SentenceTransformer", lambda _: model)
    monkeypatch.setattr(vectordb, "create_collection", lambda *_: collection)
    monkeypatch.setattr(
        vectordb,
        "query_collection",
        lambda *_: [
            {
                "id": "document-0",
                "document": "matching document",
                "cosine_similarity": 0.95,
            }
        ],
    )

    vectordb.main()

    output = capsys.readouterr().out
    assert "Stored 3 documents" in output
    assert "Document ID: document-0" in output
    assert "Cosine similarity: 0.950" in output
    assert "Document: matching document" in output
