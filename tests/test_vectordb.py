import json
import zlib
from types import SimpleNamespace

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings

from src import vectordb


def _rows(n=10):
    return [
        {
            "id": f"q-{i}",
            "question": f"Question {i}?",
            "answer": f"Answer {i}",
            "category": "c",
            "chain": "Restate: long chain text",
        }
        for i in range(n)
    ]


def _write(tmp_path, rows):
    path = tmp_path / "test_set.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_load_questions_keeps_only_id_question_and_answer_in_file_order(tmp_path):
    loaded = vectordb.load_questions(_write(tmp_path, _rows()))
    assert len(loaded) == 10
    assert loaded[0] == {"id": "q-0", "question": "Question 0?", "answer": "Answer 0"}
    assert [q["id"] for q in loaded] == [f"q-{i}" for i in range(10)]


def test_load_questions_rejects_a_row_count_other_than_ten(tmp_path):
    for n in (9, 11):
        with pytest.raises(ValueError, match="10"):
            vectordb.load_questions(_write(tmp_path, _rows(n)))


def test_load_questions_rejects_duplicate_ids(tmp_path):
    rows = _rows()
    rows[3]["id"] = rows[2]["id"]
    with pytest.raises(ValueError, match="q-2"):
        vectordb.load_questions(_write(tmp_path, rows))


@pytest.mark.parametrize("field", ["id", "question", "answer"])
@pytest.mark.parametrize("empty", ["", "   "])
def test_load_questions_rejects_an_empty_field(tmp_path, field, empty):
    rows = _rows()
    rows[4][field] = empty
    with pytest.raises(ValueError, match=field):
        vectordb.load_questions(_write(tmp_path, rows))


# ---- documents ----
def test_doc_prompt_has_the_question_and_the_answer():
    prompt = vectordb.doc_prompt({"question": "Q?", "answer": "A!"})
    assert "Q?" in prompt
    assert "A!" in prompt


def test_generate_docs_skips_stored_ids_and_a_rerun_makes_zero_calls(
    tmp_path, monkeypatch
):
    questions = vectordb.load_questions(_write(tmp_path, _rows()))
    calls = []

    def fake(prompt, model, temperature):
        calls.append(prompt)
        return "x" * 700

    monkeypatch.setattr(vectordb.gemini, "_generate", fake)
    path = tmp_path / "docs.json"
    first = vectordb.generate_docs(questions, path)
    assert len(calls) == 10
    assert vectordb.generate_docs(questions, path) == first
    assert len(calls) == 10


@pytest.mark.parametrize("text", ["", "short", "x" * 5000])
def test_validate_docs_rejects_bad_lengths(text):
    questions = [{"id": "a"}]
    with pytest.raises(ValueError, match="'a'"):
        vectordb.validate_docs(questions, {"a": text})


def test_validate_docs_rejects_mismatched_ids():
    with pytest.raises(ValueError, match="ids"):
        vectordb.validate_docs([{"id": "a"}], {"b": "x" * 700})


# ---- embeddings ----
@pytest.mark.parametrize("dim", [3072, 1536, 768, 256, 128, 64, 32])
def test_truncate_gives_the_right_length_and_unit_norm(dim):
    vector = [float(i % 7 + 1) for i in range(3072)]
    out = vectordb.truncate(vector, dim)
    assert len(out) == dim
    assert sum(x * x for x in out) == pytest.approx(1.0)


def test_truncate_of_a_zero_vector_does_not_divide_by_zero():
    assert vectordb.truncate([0.0] * 3072, 32) == [0.0] * 32


def test_truncate_rejects_sizes_above_3072():
    with pytest.raises(ValueError, match="3072"):
        vectordb.truncate([1.0] * 3072, 4096)


def test_embeddings_use_the_document_and_query_task_types(monkeypatch):
    seen = []

    class Models:
        def embed_content(self, model, contents, config):
            seen.append(config.task_type)
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[3.0, 4.0])])

    monkeypatch.setattr(
        vectordb.gemini, "get_client", lambda: SimpleNamespace(models=Models())
    )
    vectordb._embed.cache_clear()
    embeddings = vectordb.GeminiEmbeddings(32)
    assert embeddings.embed_documents(["a"]) == [[0.6, 0.8]]
    assert embeddings.embed_query("b") == [0.6, 0.8]
    assert seen == ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]


# ---- chroma store (fake embeddings, tmp_path only) ----
class FakeEmbeddings(Embeddings):
    def _vec(self, text):
        return np.random.default_rng(zlib.crc32(text.encode())).normal(size=8).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


DOCS = {f"q-{i}": f"document number {i}" for i in range(10)}


def test_open_store_rejects_a_dimension_that_is_not_in_the_list(tmp_path):
    with pytest.raises(ValueError, match="dim"):
        vectordb.open_store(100, tmp_path)


def test_the_collection_is_named_for_its_dimension(tmp_path):
    store = vectordb.build_store(768, DOCS, tmp_path, FakeEmbeddings())
    assert store._collection.name == "docs_768"


def test_building_twice_keeps_the_count_and_a_reload_matches(tmp_path):
    vectordb.build_store(32, DOCS, tmp_path, FakeEmbeddings())
    vectordb.build_store(32, DOCS, tmp_path, FakeEmbeddings())
    reloaded = vectordb.open_store(32, tmp_path, FakeEmbeddings())
    stored = reloaded.get()
    assert sorted(stored["ids"]) == sorted(DOCS)
    assert sorted(stored["documents"]) == sorted(DOCS.values())


def test_the_space_is_cosine_not_euclidean(tmp_path):
    class Scaled(Embeddings):
        def embed_documents(self, texts):
            return [[3.0, 0.0]] * len(texts)

        def embed_query(self, text):
            return [1.0, 0.0]

    store = vectordb.build_store(32, {"a": "doc"}, tmp_path, Scaled())
    _, _, distance = vectordb.top1(store, "anything")
    assert distance == pytest.approx(0.0, abs=1e-6)


def test_top1_matches_a_brute_force_numpy_cosine_search(tmp_path):
    fake = FakeEmbeddings()
    store = vectordb.build_store(32, DOCS, tmp_path, fake)
    matrix = np.array(fake.embed_documents(list(DOCS.values())))
    for i in range(10):
        query = f"question about thing {i}"
        q = np.array(fake.embed_query(query))
        sims = matrix @ q / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q))
        doc_id, _, distance = vectordb.top1(store, query)
        assert doc_id == list(DOCS)[int(np.argmax(sims))]
        assert distance == pytest.approx(1 - sims.max(), abs=1e-6)


def test_check_dim_and_smallest_perfect_on_a_hand_built_case(tmp_path):
    questions = [{"id": f"q-{i}", "question": DOCS[f"q-{i}"]} for i in range(10)]
    store = vectordb.build_store(32, DOCS, tmp_path, FakeEmbeddings())
    rows = vectordb.check_dim(store, questions)
    assert all(r["correct"] for r in rows)
    assert vectordb.smallest_perfect({64: rows, 32: rows, 128: rows}) == 32
    wrong = [{**rows[0], "correct": False}]
    assert vectordb.smallest_perfect({32: wrong}) is None
    questions[0]["question"] = DOCS["q-1"]
    miss = vectordb.check_dim(store, questions)[0]
    assert miss["correct"] is False
    assert miss["winner"] == "q-1"


def test_sweep_times_each_dimension_and_embeds_the_questions_before_timing(
    tmp_path, monkeypatch
):
    calls = []

    def fake_embed(text, task_type):
        calls.append(task_type)
        rng = np.random.default_rng(zlib.crc32(text.encode()))
        return tuple(rng.normal(size=3072))

    monkeypatch.setattr(vectordb, "_embed", fake_embed)
    questions = [{"id": f"q-{i}", "question": DOCS[f"q-{i}"]} for i in range(10)]
    results, seconds = vectordb.sweep(questions, DOCS, tmp_path)
    assert list(results) == list(seconds) == vectordb.config.DIMS
    assert all(s >= 0 for s in seconds.values())
    assert calls[:10] == ["RETRIEVAL_QUERY"] * 10
