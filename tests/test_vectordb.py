"""
Tests for src/vectordb.py.

Most tests use Chroma's in-memory EphemeralClient with a simple fake
embedding function, rather than downloading the real sentence-transformer
model or a GPU-based LLM, keeping tests fast and runnable in CI without
network access or a GPU.
"""

import json
import uuid

import chromadb

from src.vectordb import (
    TRICK_QUESTIONS,
    extract_dialogue_lines,
    load_dialogue_into_collection,
    load_trick_questions_into_collection,
    query_collection,
    save_rag_results,
)

# ---------------------------------------------------------------------------
# FAKE EMBEDDING FUNCTION
# A simple, deterministic embedding function that doesn't require
# downloading any model, used for all in-memory Chroma tests.
# ---------------------------------------------------------------------------


class FakeEmbeddingFunction:
    """
    Produces a short, deterministic vector for each text based on its
    length and character codes, just consistent enough that identical or
    similar texts produce similar vectors, without needing a real model.
    """

    def __call__(self, input):
        return [self._embed_one(text) for text in input]

    def embed_query(self, input):
        return [self._embed_one(text) for text in input]

    def _embed_one(self, text):
        return [float(len(text)), float(sum(ord(c) for c in text) % 1000)]

    def name(self):
        return "fake_embedding_function"


def make_test_collection():
    """Create a fresh, in-memory Chroma collection for testing, with a unique
    name each time to avoid state leaking between tests."""
    client = chromadb.EphemeralClient()
    collection_name = f"test_collection_{uuid.uuid4().hex}"
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=FakeEmbeddingFunction(),
    )


# ---------------------------------------------------------------------------
# extract_dialogue_lines() TESTS
# Mocked via a temp file, so this doesn't depend on the real dataset existing.
# ---------------------------------------------------------------------------


def test_extract_dialogue_lines_reads_all_lines(tmp_path):
    """Every turn across every scene should be extracted as a separate line."""
    fake_scenes = [
        {
            "movie": "A New Hope",
            "turns": [
                {"speaker": "Luke", "text": "Hello there."},
                {"speaker": "Leia", "text": "Help me, Obi-Wan."},
            ],
        },
        {"movie": "A New Hope", "turns": [{"speaker": "Han", "text": "I know."}]},
    ]
    fake_file = tmp_path / "fake_scenes.jsonl"
    fake_file.write_text(
        "\n".join(json.dumps(s) for s in fake_scenes), encoding="utf-8"
    )

    lines = extract_dialogue_lines(str(fake_file), max_lines=100)

    assert len(lines) == 3
    assert lines[0] == {"speaker": "Luke", "text": "Hello there."}


def test_extract_dialogue_lines_respects_max_lines(tmp_path):
    """extract_dialogue_lines should stop once max_lines is reached."""
    fake_scenes = [
        {
            "movie": "A New Hope",
            "turns": [{"speaker": "Luke", "text": f"Line {i}"} for i in range(10)],
        },
    ]
    fake_file = tmp_path / "fake_scenes.jsonl"
    fake_file.write_text(json.dumps(fake_scenes[0]), encoding="utf-8")

    lines = extract_dialogue_lines(str(fake_file), max_lines=3)

    assert len(lines) == 3


def test_extract_dialogue_lines_empty_file_returns_empty_list(tmp_path):
    """An empty dataset file should produce an empty list, not an error."""
    fake_file = tmp_path / "empty.jsonl"
    fake_file.write_text("", encoding="utf-8")

    lines = extract_dialogue_lines(str(fake_file))

    assert lines == []


# ---------------------------------------------------------------------------
# load_dialogue_into_collection() TESTS
# ---------------------------------------------------------------------------


def test_load_dialogue_into_collection_adds_all_lines():
    """Every line should be added as a separate document in the collection."""
    collection = make_test_collection()
    lines = [
        {"speaker": "Luke", "text": "Hello there."},
        {"speaker": "Leia", "text": "Help me, Obi-Wan."},
    ]

    load_dialogue_into_collection(collection, lines)

    assert collection.count() == 2


def test_load_dialogue_into_collection_formats_documents_with_speaker():
    """Each stored document should include the speaker name and line text."""
    collection = make_test_collection()
    lines = [{"speaker": "Yoda", "text": "Do or do not."}]

    load_dialogue_into_collection(collection, lines)

    stored = collection.get()
    assert "Yoda: Do or do not." in stored["documents"]


# ---------------------------------------------------------------------------
# load_trick_questions_into_collection() TESTS
# ---------------------------------------------------------------------------


def test_load_trick_questions_into_collection_adds_all_ten():
    """All 10 trick questions should be added to the collection."""
    collection = make_test_collection()

    load_trick_questions_into_collection(collection)

    assert collection.count() == len(TRICK_QUESTIONS)


def test_load_trick_questions_into_collection_includes_question_and_answer():
    """Each stored document should contain both the question and its answer."""
    collection = make_test_collection()

    load_trick_questions_into_collection(collection)

    stored = collection.get()
    first_doc = stored["documents"][0]
    assert "Question:" in first_doc
    assert "Answer:" in first_doc


# ---------------------------------------------------------------------------
# query_collection() TESTS
# ---------------------------------------------------------------------------


def test_query_collection_returns_requested_number_of_results():
    """query_collection should return exactly n_results matches."""
    collection = make_test_collection()
    lines = [
        {"speaker": "Luke", "text": "Hello there."},
        {"speaker": "Leia", "text": "Help me, Obi-Wan."},
        {"speaker": "Han", "text": "I know."},
    ]
    load_dialogue_into_collection(collection, lines)

    results = query_collection(collection, "Hello there.", n_results=2)

    assert len(results["documents"][0]) == 2


def test_query_collection_finds_exact_match_first():
    """Querying with text identical to a stored document should return that document first."""
    collection = make_test_collection()
    lines = [
        {"speaker": "Luke", "text": "Hello there."},
        {"speaker": "Leia", "text": "Completely different unrelated text."},
    ]
    load_dialogue_into_collection(collection, lines)

    results = query_collection(collection, "Luke: Hello there.", n_results=1)

    assert results["documents"][0][0] == "Luke: Hello there."


# ---------------------------------------------------------------------------
# save_rag_results() TESTS
# ---------------------------------------------------------------------------


def test_save_rag_results_writes_valid_json(tmp_path):
    """save_rag_results should write a valid, readable JSON file."""
    rag_results = [
        {"question": "Q1", "rag_answer": "A1", "expected": "A1"},
        {"question": "Q2", "rag_answer": "A2 wrong", "expected": "A2"},
    ]
    output_path = tmp_path / "results.json"

    save_rag_results(rag_results, output_path=str(output_path))

    with open(output_path, encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded == rag_results


# ---------------------------------------------------------------------------
# load_base_model() / rag_answer() / run_rag_test() -- GPU-dependent
# Intentionally NOT tested here, since they require a GPU and downloading
# a real model, which this CI environment does not have.
# ---------------------------------------------------------------------------


def test_vectordb_module_is_importable_without_a_gpu():
    """
    Confirm that importing src.vectordb doesn't itself require a GPU or
    trigger model loading -- only calling load_base_model()/rag_answer()
    should need one.
    """
    import src.vectordb  # noqa: F401
