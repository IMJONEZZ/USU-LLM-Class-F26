import hashlib
import json
import math
import types

import chromadb
import pytest

from src import vectordb

DIMENSIONS = 64


def fake_embed(texts):
    """Deterministic bag-of-words vectors: shared words mean closer vectors."""
    vectors = []
    for text in texts:
        vector = [0.0] * DIMENSIONS
        for word in text.lower().replace("?", " ").replace(".", " ").split():
            digest = hashlib.sha256(word.encode()).digest()
            vector[digest[0] % DIMENSIONS] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        vectors.append([x / norm for x in vector])
    return vectors


PARAGRAPHS = [
    ("Paris is the capital of France and sits on the Seine river.", "Where is Paris"),
    (
        "Mount Everest is the tallest mountain above sea level.",
        "Which mountain is tallest",
    ),
    (
        "Photosynthesis lets plants turn sunlight into chemical energy.",
        "How do plants use sunlight",
    ),
    (
        "The violin is a string instrument played with a bow.",
        "Which instrument uses a bow",
    ),
]


def squad_file(tmp_path, paragraphs=PARAGRAPHS, repeat_first=True):
    """Write a tiny SQuAD-shaped file; the first paragraph is repeated on purpose."""
    entries = [
        {"context": context, "qas": [{"question": question}, {"question": "extra"}]}
        for context, question in paragraphs
    ]
    if repeat_first:
        entries.append({"context": paragraphs[0][0], "qas": [{"question": "dup"}]})
    entries.append({"context": "A paragraph with no questions.", "qas": []})
    path = tmp_path / "squad.json"
    path.write_text(json.dumps({"data": [{"paragraphs": entries}]}), encoding="utf-8")
    return path


@pytest.fixture
def client(tmp_path):
    return chromadb.PersistentClient(path=str(tmp_path / "db"))


@pytest.fixture
def items(tmp_path):
    return vectordb.load_squad(squad_file(tmp_path))


@pytest.fixture
def collection(client, items):
    collection, _ = vectordb.build_collection(client, items, fake_embed)
    return collection


def word_count(text):
    return len(text.split())


# --- loading and deduplication -------------------------------------------------


def test_load_squad_removes_duplicates_and_keeps_file_order(items):
    assert [item["id"] for item in items] == ["p0000", "p0001", "p0002", "p0003"]
    assert items[0]["context"] == PARAGRAPHS[0][0]


def test_load_squad_uses_first_question_and_drops_paragraphs_without_questions(items):
    assert items[0]["question"] == "Where is Paris"
    assert all("no questions" not in item["context"] for item in items)


def test_download_squad_keeps_existing_local_copy(tmp_path, monkeypatch):
    path = tmp_path / "dev.json"
    path.write_text("{}")

    def fail(*args, **kwargs):
        raise AssertionError("should not download")

    monkeypatch.setattr(vectordb.urllib.request, "urlopen", fail)
    assert vectordb.download_squad(path) == path


def test_download_squad_writes_the_file_once(tmp_path, monkeypatch):
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"data": []}'

    def fake_urlopen(url, timeout):
        calls.append(url)
        return FakeResponse()

    monkeypatch.setattr(vectordb.urllib.request, "urlopen", fake_urlopen)
    path = tmp_path / "nested" / "dev.json"
    vectordb.download_squad(path, url="http://example.test/squad.json")
    vectordb.download_squad(path, url="http://example.test/squad.json")
    assert path.read_bytes() == b'{"data": []}'
    assert calls == ["http://example.test/squad.json"]


# --- the plan's rule for over-long paragraphs ------------------------------------


def make_items(lengths):
    return [
        {"id": f"p{i:04d}", "context": "word " * length, "question": "q"}
        for i, length in enumerate(lengths)
    ]


def test_select_keeps_everything_when_few_paragraphs_are_too_long():
    # 1 of 20 is over the limit: exactly 5%, so keep and report.
    lengths = [5] * 19 + [50]
    selected, report = vectordb.select_paragraphs(
        make_items(lengths), 20, 10, word_count
    )
    assert len(selected) == 20
    assert report == {"limit": 10, "checked": 20, "over_limit": 1, "skipped": 0}


def test_select_skips_long_paragraphs_when_share_is_above_five_percent():
    # 2 of 20 over the limit (10%): skip them and take the next ones in order.
    lengths = [5, 50, 5, 5, 50] + [5] * 20
    selected, report = vectordb.select_paragraphs(
        make_items(lengths), 20, 10, word_count
    )
    assert len(selected) == 20
    assert all(word_count(item["context"]) <= 10 for item in selected)
    assert [item["id"] for item in selected][:3] == ["p0000", "p0002", "p0003"]
    assert report["over_limit"] == 2
    assert report["skipped"] == 2


def test_select_rejects_zero_paragraphs():
    with pytest.raises(ValueError):
        vectordb.select_paragraphs(make_items([5]), 0, 10, word_count)


def test_select_returns_what_exists_when_fewer_than_n_paragraphs():
    selected, report = vectordb.select_paragraphs(
        make_items([5, 5]), 10, 10, word_count
    )
    assert len(selected) == 2
    assert report["checked"] == 2


# --- building the collection ---------------------------------------------------


def test_collection_uses_cosine_distance(collection):
    # Chroma defaults to L2, so this setting must be explicit.
    assert collection.configuration["hnsw"]["space"] == "cosine"


def test_build_collection_adds_every_paragraph(client, items):
    collection, added = vectordb.build_collection(client, items, fake_embed)
    assert added == 4
    assert collection.count() == 4


def test_build_collection_does_not_embed_stored_paragraphs_again(client, items):
    vectordb.build_collection(client, items, fake_embed)
    seen = []

    def counting_embed(texts):
        seen.extend(texts)
        return fake_embed(texts)

    collection, added = vectordb.build_collection(client, items, counting_embed)
    assert added == 0
    assert seen == []
    assert collection.count() == 4


def test_build_collection_adds_only_the_new_paragraphs(client, items):
    vectordb.build_collection(client, items[:2], fake_embed)
    collection, added = vectordb.build_collection(client, items, fake_embed)
    assert added == 2
    assert collection.count() == 4


def test_build_collection_rejects_empty_input(client):
    with pytest.raises(ValueError):
        vectordb.build_collection(client, [], fake_embed)


def test_database_reloads_from_disk(tmp_path, items):
    path = str(tmp_path / "reload_db")
    first = chromadb.PersistentClient(path=path)
    collection, _ = vectordb.build_collection(first, items, fake_embed)
    before = vectordb.query(collection, "Where is Paris", fake_embed, 3)
    del collection, first

    second = chromadb.PersistentClient(path=path)
    reopened, added = vectordb.build_collection(second, items, fake_embed)
    after = vectordb.query(reopened, "Where is Paris", fake_embed, 3)
    assert added == 0
    assert [r["id"] for r in after] == [r["id"] for r in before]
    assert after[0]["id"] == "p0000"


# --- querying ----------------------------------------------------------------


def test_query_returns_nearest_paragraph_first(collection):
    results = vectordb.query(collection, "which instrument uses a bow", fake_embed, 3)
    assert results[0]["id"] == "p0003"
    assert results[0]["document"] == PARAGRAPHS[3][0]


def test_query_orders_results_by_distance(collection):
    results = vectordb.query(collection, "plants and sunlight", fake_embed, 4)
    distances = [r["distance"] for r in results]
    assert distances == sorted(distances)


def test_query_returns_at_most_k_results(collection):
    assert len(vectordb.query(collection, "mountain", fake_embed, 2)) == 2


def test_query_with_k_larger_than_the_database_returns_everything(collection):
    assert len(vectordb.query(collection, "mountain", fake_embed, 50)) == 4


def test_query_on_empty_database_returns_nothing(client):
    empty = client.get_or_create_collection("empty", embedding_function=None)
    assert vectordb.query(empty, "anything", fake_embed) == []


@pytest.mark.parametrize("text", ["", "   "])
def test_query_rejects_blank_text(collection, text):
    with pytest.raises(ValueError):
        vectordb.query(collection, text, fake_embed)


def test_query_rejects_k_below_one(collection):
    with pytest.raises(ValueError):
        vectordb.query(collection, "mountain", fake_embed, 0)


# --- measuring hit@k and latency -------------------------------------------------


def test_evaluate_finds_source_paragraphs(collection, items):
    metrics = vectordb.evaluate(collection, items, fake_embed)
    assert metrics["queries"] == 4
    assert metrics["hit_rates"][5] == 1.0
    assert metrics["hit_rates"][1] <= metrics["hit_rates"][3] <= metrics["hit_rates"][5]


def test_evaluate_counts_a_miss(collection, items):
    wrong = [dict(item) for item in items]
    wrong[0]["question"] = "Which instrument uses a bow"  # belongs to p0003
    metrics = vectordb.evaluate(collection, wrong, fake_embed, ks=(1,))
    assert metrics["hit_rates"][1] == pytest.approx(0.75)


def test_evaluate_reports_latencies(collection, items):
    metrics = vectordb.evaluate(collection, items, fake_embed)
    assert 0 <= metrics["median_latency_s"] <= metrics["max_latency_s"]
    assert metrics["first_latency_s"] >= 0


def test_evaluate_rejects_empty_items(collection):
    with pytest.raises(ValueError):
        vectordb.evaluate(collection, [], fake_embed)


# --- report text ---------------------------------------------------------------


def test_format_report_marks_pass_and_miss(collection, items):
    metrics = {
        "queries": 200,
        "hit_rates": {1: 0.7, 3: 0.7, 5: 0.9},
        "median_latency_s": 0.4,
        "first_latency_s": 0.5,
        "max_latency_s": 0.6,
    }
    selection = {"limit": 256, "checked": 200, "over_limit": 3, "skipped": 0}
    text = vectordb.format_report(selection, 200, 200, metrics, ["example"])
    assert "hit@1: 70.0% (target 60%) PASS" in text
    assert "hit@3: 70.0% (target 75%) MISS" in text
    assert "median query latency: 400 ms (target under 250 ms) MISS" in text
    assert "3 of 200 paragraphs over the limit" in text


def test_describe_example_shows_expected_paragraph_and_hit(collection, items):
    text = vectordb.describe_example(collection, items[0], fake_embed)
    assert "expected p0000" in text
    assert text.endswith("HIT")


# --- default embedder and command line, with the real model replaced -----------


class FakeTokenizer:
    def __init__(self):
        self.truncation = {"max_length": 256}

    def no_truncation(self):
        pass

    def no_padding(self):
        pass

    def encode(self, text):
        return types.SimpleNamespace(ids=text.split())


class FakeOnnx:
    DOWNLOAD_PATH = __import__("pathlib").Path("/fake")
    EXTRACTED_FOLDER_NAME = "onnx"
    Tokenizer = types.SimpleNamespace(from_file=lambda path: FakeTokenizer())
    tokenizer = FakeTokenizer()

    def __call__(self, texts):
        return fake_embed(texts)


def test_default_embedder_returns_embed_counter_and_limit(monkeypatch):
    monkeypatch.setattr(vectordb, "ONNXMiniLM_L6_V2", FakeOnnx)
    embed, count_tokens, limit = vectordb.default_embedder()
    vectors = embed(["hello world"])
    assert len(vectors) == 1 and len(vectors[0]) == DIMENSIONS
    assert all(isinstance(x, float) for x in vectors[0])
    assert count_tokens("one two three") == 3
    assert limit == 256


def test_main_builds_queries_and_prints_a_report(tmp_path, monkeypatch, capsys):
    data = squad_file(tmp_path)
    monkeypatch.setattr(
        vectordb, "default_embedder", lambda: (fake_embed, word_count, 256)
    )
    monkeypatch.setattr(vectordb, "download_squad", lambda path: data)
    args = [
        "--data-dir",
        str(tmp_path),
        "--db-path",
        str(tmp_path / "main_db"),
        "--n",
        "4",
    ]

    assert vectordb.main(args) == 0
    first = capsys.readouterr().out
    assert "Paragraphs: 4 in the database (4 newly embedded)" in first
    assert "hit@1" in first and "hit@5" in first

    assert vectordb.main(args) == 0
    second = capsys.readouterr().out
    assert "(0 newly embedded)" in second
