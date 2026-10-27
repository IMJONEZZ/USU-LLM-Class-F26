"""Self-contained ChromaDB tests: real client, temp storage, fixed vectors.

No downloads, GPUs, APIs, pretrained models, or repository data/ files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src import vectordb

# Orthogonal unit vectors — deterministic and independent of any embedding model.
DOC_EMBEDDINGS = {
    "doc-alpha": [1.0, 0.0, 0.0],
    "doc-beta": [0.0, 1.0, 0.0],
    "doc-gamma": [0.0, 0.0, 1.0],
}

SAMPLE_RECORDS: list[vectordb.QARecord] = [
    {
        "id": "doc-alpha",
        "question": "What color is the sky on a clear day?",
        "answer": "Blue",
        "assumptions": "daytime clear weather",
    },
    {
        "id": "doc-beta",
        "question": "How many legs does a typical dog have?",
        "answer": "Four",
        "assumptions": "typical healthy dog",
    },
    {
        "id": "doc-gamma",
        "question": "What gas do plants absorb during photosynthesis?",
        "answer": "Carbon dioxide",
        "assumptions": "standard plant photosynthesis",
    },
]


def _embeddings_for(records: list[vectordb.QARecord]) -> list[list[float]]:
    return [DOC_EMBEDDINGS[record["id"]] for record in records]


def _write_jsonl(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def sample_jsonl(tmp_path: Path) -> Path:
    lines = [
        (
            '{"id": "doc-alpha", "question": "What color is the sky on a clear day?", '
            '"answer": "Blue", "assumptions": "daytime clear weather"}'
        ),
        (
            '{"id": "doc-beta", "question": "How many legs does a typical dog have?", '
            '"answer": "Four", "assumptions": "typical healthy dog"}'
        ),
        (
            '{"id": "doc-gamma", "question": "What gas do plants absorb during photosynthesis?", '
            '"answer": "Carbon dioxide", "assumptions": "standard plant photosynthesis"}'
        ),
    ]
    return _write_jsonl(tmp_path / "documents.jsonl", lines)


@pytest.fixture
def populated(tmp_path: Path):
    """Real PersistentClient with supplied numerical embeddings (no model)."""
    persist_dir = tmp_path / "chroma"
    client, collection, records = vectordb.build_collection(
        persist_directory=persist_dir,
        documents=SAMPLE_RECORDS,
        embeddings=_embeddings_for(SAMPLE_RECORDS),
    )
    return client, collection, records, persist_dir


def test_load_documents_from_temp_jsonl(sample_jsonl: Path):
    records = vectordb.load_documents(sample_jsonl)
    assert len(records) == 3
    assert records[0]["id"] == "doc-alpha"
    assert records[0]["answer"] == "Blue"
    assert "sky" in records[0]["question"]


def test_load_documents_skips_blank_lines(tmp_path: Path):
    path = _write_jsonl(
        tmp_path / "blank.jsonl",
        [
            '{"id": "doc-alpha", "question": "q1", "answer": "a1"}',
            "",
            '{"id": "doc-beta", "question": "q2", "answer": "a2"}',
        ],
    )
    assert len(vectordb.load_documents(path)) == 2


def test_load_paraphrases_missing_and_blank_lines(tmp_path: Path):
    assert vectordb.load_paraphrases(tmp_path / "missing.jsonl") == []
    path = _write_jsonl(
        tmp_path / "paraphrases.jsonl",
        [
            '{"id": "p1", "expected_id": "doc-alpha", "question": "sky color?"}',
            "",
            '{"id": "p2", "expected_id": "doc-beta", "question": "dog legs?"}',
        ],
    )
    paraphrases = vectordb.load_paraphrases(path)
    assert len(paraphrases) == 2
    assert paraphrases[0]["expected_id"] == "doc-alpha"


def test_collection_creation_and_get_or_create(tmp_path: Path):
    client = vectordb.create_persistent_client(tmp_path / "chroma")
    collection = vectordb.get_or_create_docs_collection(client)
    assert collection.name == vectordb.COLLECTION_NAME
    assert collection.count() == 0
    again = vectordb.get_or_create_docs_collection(client)
    assert again.name == collection.name


def test_insertion_count_and_metadata(populated):
    _client, collection, records, _persist_dir = populated
    assert collection.count() == len(records) == 3
    stored = collection.get(include=["documents", "metadatas"])
    assert set(stored["ids"]) == {record["id"] for record in records}
    by_id = {
        item_id: meta
        for item_id, meta in zip(stored["ids"], stored["metadatas"], strict=True)
    }
    assert by_id["doc-beta"]["answer"] == "Four"


def test_query_with_numerical_embeddings(populated):
    _client, collection, _records, _persist_dir = populated
    results = vectordb.query_similar(
        collection,
        query_embeddings=[[0.0, 0.95, 0.05]],
        n_results=1,
    )
    assert results["ids"][0][0] == "doc-beta"
    assert results["metadatas"][0][0]["answer"] == "Four"

    hits = vectordb.retrieve_answers(
        collection,
        questions=["unused when embeddings are supplied"],
        n_results=1,
        query_embeddings=[[0.02, 0.02, 0.96]],
    )
    assert hits[0][0]["id"] == "doc-gamma"
    assert hits[0][0]["answer"] == "Carbon dioxide"
    assert hits[0][0]["distance"] is not None


def test_query_requires_input(populated):
    _client, collection, _records, _persist_dir = populated
    with pytest.raises(ValueError, match="query_texts or query_embeddings"):
        vectordb.query_similar(collection)


def test_idempotent_upsert_does_not_duplicate(populated):
    _client, collection, records, persist_dir = populated
    assert collection.count() == 3
    vectordb.build_collection(
        persist_directory=persist_dir,
        documents=records,
        embeddings=_embeddings_for(records),
    )
    assert collection.count() == 3
    vectordb.upsert_documents(collection, records, embeddings=_embeddings_for(records))
    assert collection.count() == 3


def test_persistence_across_clients(tmp_path: Path):
    persist_dir = tmp_path / "chroma-persist"
    vectordb.build_collection(
        persist_directory=persist_dir,
        documents=SAMPLE_RECORDS,
        embeddings=_embeddings_for(SAMPLE_RECORDS),
    )

    client = vectordb.create_persistent_client(persist_dir)
    collection = vectordb.get_or_create_docs_collection(client)
    assert collection.count() == 3
    hits = vectordb.retrieve_answers(
        collection,
        questions=["probe"],
        n_results=1,
        query_embeddings=[[1.0, 0.0, 0.0]],
    )
    assert hits[0][0]["id"] == "doc-alpha"
    assert hits[0][0]["answer"] == "Blue"


def test_evaluate_exact_and_paraphrase_scoring(populated, tmp_path: Path):
    _client, collection, records, _persist_dir = populated
    query_embeddings = _embeddings_for(records)
    exact = vectordb.evaluate_exact_questions(
        collection, records, query_embeddings=query_embeddings
    )
    assert exact["total"] == 3
    assert exact["correct"] == 3
    assert exact["success_rate"] == 1.0

    paraphrase_path = _write_jsonl(
        tmp_path / "paraphrases.jsonl",
        [
            '{"id": "p-alpha", "expected_id": "doc-alpha", "question": "sky color?"}',
            '{"id": "p-beta", "expected_id": "doc-beta", "question": "dog legs?"}',
        ],
    )
    paraphrases = vectordb.load_paraphrases(paraphrase_path)
    # Same deterministic vectors keyed by expected target id.
    para_vectors = [DOC_EMBEDDINGS[item["expected_id"]] for item in paraphrases]
    para = vectordb.evaluate_paraphrases(
        collection, paraphrases, query_embeddings=para_vectors
    )
    assert para["correct"] == 2
    assert para["total"] == 2


def test_score_retrieval_partial_mismatch():
    hits = [
        [{"id": "doc-alpha", "question": "q", "answer": "a", "distance": 0.1}],
        [{"id": "wrong", "question": "q", "answer": "a", "distance": 0.2}],
        [],
    ]
    score = vectordb.score_retrieval(
        ["q1", "q2", "q3"],
        ["doc-alpha", "doc-beta", "doc-gamma"],
        hits,
    )
    assert score["correct"] == 1
    assert score["total"] == 3
    assert score["success_rate"] == pytest.approx(1 / 3)
    assert score["details"][2]["retrieved_id"] is None


def test_format_helpers():
    empty = vectordb.format_query_results(
        ["missing"],
        {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]},
    )
    assert "Query: missing" in empty
    assert "(no results)" in empty

    report = vectordb.format_retrieval_report(
        ["sky?"],
        [
            [
                {
                    "id": "doc-alpha",
                    "question": "What color is the sky on a clear day?",
                    "answer": "Blue",
                    "distance": 0.0123,
                }
            ]
        ],
    )
    assert "id=doc-alpha" in report
    assert "distance=0.0123" in report
    assert "answer: Blue" in report


def test_main_exact_benchmark_with_embeddings(tmp_path: Path, capsys):
    """Exact benchmark via supplied vectors; paraphrases covered in a separate test."""
    exit_code = vectordb.main(
        persist_directory=tmp_path / "chroma-main",
        documents=SAMPLE_RECORDS,
        paraphrase_path=tmp_path / "missing-paraphrases.jsonl",
        embeddings=_embeddings_for(SAMPLE_RECORDS),
        query_embeddings=_embeddings_for(SAMPLE_RECORDS),
        n_results=1,
        run_paraphrases=True,
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Stored documents: 3" in output
    assert "Exact retrieval success: 3/3" in output
    assert "answer: Blue" in output
    assert "Paraphrase retrieval success" not in output


def test_main_custom_queries_with_embeddings(tmp_path: Path, capsys):
    exit_code = vectordb.main(
        persist_directory=tmp_path / "chroma-queries",
        documents=SAMPLE_RECORDS,
        queries=["plant gas"],
        embeddings=_embeddings_for(SAMPLE_RECORDS),
        query_embeddings=[[0.0, 0.0, 1.0]],
        n_results=1,
        run_paraphrases=False,
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Stored documents: 3" in output
    assert "doc-gamma" in output
    assert "Carbon dioxide" in output


def test_main_paraphrase_section_offline(tmp_path: Path, capsys):
    """Cover paraphrase printing without invoking the default embedding model.

    main() paraphrases use query_texts; supply a collection built with a custom
    embedding function that returns our fixed numerical vectors.
    """
    from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

    class FixedEmbeddingFunction(EmbeddingFunction[Documents]):
        def __init__(self) -> None:
            pass

        @staticmethod
        def name() -> str:
            return "fixed_test_embeddings"

        def get_config(self) -> dict[str, str]:
            return {}

        @staticmethod
        def build_from_config(config: dict[str, str]) -> FixedEmbeddingFunction:
            return FixedEmbeddingFunction()

        def __call__(self, input: Documents) -> Embeddings:
            # Map known phrases onto the orthogonal vectors used above.
            out: list[list[float]] = []
            for text in input:
                lower = text.lower()
                if "sky" in lower or "color" in lower or "clear day" in lower:
                    out.append(DOC_EMBEDDINGS["doc-alpha"])
                elif "dog" in lower or "legs" in lower:
                    out.append(DOC_EMBEDDINGS["doc-beta"])
                elif "plant" in lower or "gas" in lower or "photosynthesis" in lower:
                    out.append(DOC_EMBEDDINGS["doc-gamma"])
                else:
                    out.append([0.0, 0.0, 0.0])
            return out

    paraphrase_path = _write_jsonl(
        tmp_path / "paraphrases.jsonl",
        [
            '{"id": "p-alpha", "expected_id": "doc-alpha", "question": "sky color?"}',
            '{"id": "p-gamma", "expected_id": "doc-gamma", "question": "plant gas?"}',
        ],
    )
    exit_code = vectordb.main(
        persist_directory=tmp_path / "chroma-para",
        documents=SAMPLE_RECORDS,
        paraphrase_path=paraphrase_path,
        embedding_function=FixedEmbeddingFunction(),
        run_paraphrases=True,
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Exact retrieval success: 3/3" in output
    assert "Paraphrase retrieval success: 2/2" in output


def test_build_collection_from_temp_data_path(sample_jsonl: Path, tmp_path: Path):
    client, collection, records = vectordb.build_collection(
        persist_directory=tmp_path / "chroma-file",
        data_path=sample_jsonl,
        embeddings=_embeddings_for(SAMPLE_RECORDS),
    )
    assert len(records) == 3
    assert collection.count() == 3
    assert client is not None


def test_retrieve_answers_query_texts_with_fixed_ef(tmp_path: Path):
    """Cover query_texts path using a local fixed EF (still no model download)."""
    from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

    class FixedEmbeddingFunction(EmbeddingFunction[Documents]):
        def __init__(self) -> None:
            pass

        @staticmethod
        def name() -> str:
            return "fixed_test_embeddings"

        def get_config(self) -> dict[str, str]:
            return {}

        @staticmethod
        def build_from_config(config: dict[str, str]) -> FixedEmbeddingFunction:
            return FixedEmbeddingFunction()

        def __call__(self, input: Documents) -> Embeddings:
            out: list[list[float]] = []
            for text in input:
                lower = text.lower()
                if "sky" in lower or "color" in lower:
                    out.append(DOC_EMBEDDINGS["doc-alpha"])
                elif "dog" in lower or "legs" in lower:
                    out.append(DOC_EMBEDDINGS["doc-beta"])
                else:
                    out.append(DOC_EMBEDDINGS["doc-gamma"])
            return out

    _client, collection, _records = vectordb.build_collection(
        persist_directory=tmp_path / "chroma-texts",
        documents=SAMPLE_RECORDS,
        embedding_function=FixedEmbeddingFunction(),
    )
    hits = vectordb.retrieve_answers(
        collection,
        ["How many legs does a typical dog have?"],
        n_results=1,
    )
    assert hits[0][0]["id"] == "doc-beta"
