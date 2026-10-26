"""Tests for the vector database.

Nothing here downloads a model or writes a database to disk. The encoder and
the client are both arguments, so the tests hand over fakes, which matters for
two reasons. The Star Wars file is gitignored so CI has never seen it, and the
embedding model is a download that the runner shouldn't be making.

The fake encoder returns vectors I chose, so "which document comes back first"
is something I can assert rather than something I have to hope about. A test
that embeds real text and checks the result looks plausible isn't a test.
"""

import json

import pytest

from src.vectordb import (
    SCRIPT_COLLECTION,
    VECTOR_SIZE,
    build,
    create_collection,
    embed,
    load_script,
    search,
    to_documents,
    upsert_documents,
)


class FakeEncoder:
    """Maps text to a vector by counting a couple of marker words.

    Crude, but it means a query about the Force lands near documents about the
    Force and I know that before the test runs.
    """

    def __init__(self, normalize_check=None):
        self.calls = []
        self.normalize_check = normalize_check

    def encode(self, texts, normalize_embeddings=False):
        self.calls.append({"texts": list(texts), "normalize": normalize_embeddings})
        out = []
        for t in texts:
            low = t.lower()
            out.append(
                [low.count("force"), low.count("droid"), float(len(low)) / 100, 1.0]
            )
        return out


class FakeHit:
    def __init__(self, score, payload):
        self.score = score
        self.payload = payload


class FakeResult:
    def __init__(self, points):
        self.points = points


class FakeClient:
    """Records what it was told to do and returns queued hits."""

    def __init__(self, exists=False, hits=None):
        self.exists = exists
        self.hits = hits or []
        self.created = []
        self.deleted = []
        self.upserts = []
        self.queries = []

    def collection_exists(self, name):
        return self.exists

    def delete_collection(self, name):
        self.deleted.append(name)
        self.exists = False

    def create_collection(self, collection_name, vectors_config):
        self.created.append({"name": collection_name, "config": vectors_config})

    def upsert(self, collection_name, points):
        self.upserts.append({"name": collection_name, "points": points})

    def query_points(self, collection_name, query, limit, with_payload):
        self.queries.append({"name": collection_name, "query": query, "limit": limit})
        return FakeResult(self.hits[:limit])


# --- reading the script -------------------------------------------------------


@pytest.fixture
def script_file(tmp_path):
    rows = [
        {"Character": "THREEPIO", "Line": "We're doomed."},
        {"Character": "LUKE", "Line": "The Force?"},
        {"Character": "", "Line": "A line with no speaker."},
        {"Character": "HAN", "Line": ""},
        "not a dict at all",
    ]
    path = tmp_path / "SW.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return str(path)


def test_load_script_keeps_the_speaker(script_file):
    """Assignment 2's loader throws the character away. This one shouldn't."""
    records = load_script(script_file)
    assert records[0] == {"character": "THREEPIO", "line": "We're doomed."}


def test_load_script_skips_rows_with_no_line(script_file):
    # An empty Line and a bare string in the list would both become documents
    # with nothing in them, which then match everything weakly.
    records = load_script(script_file)
    assert "" not in [r["line"] for r in records]
    assert len(records) == 3


def test_to_documents_puts_the_speaker_in_the_embedded_text(script_file):
    docs = to_documents(load_script(script_file))
    assert docs[0]["text"] == "THREEPIO: We're doomed."


def test_to_documents_leaves_an_unattributed_line_alone(script_file):
    """No speaker shouldn't mean a document that starts with a stray colon."""
    docs = to_documents(load_script(script_file))
    unattributed = next(d for d in docs if d["character"] == "")
    assert unattributed["text"] == "A line with no speaker."


def test_to_documents_keeps_the_parts_separate_in_the_payload(script_file):
    docs = to_documents(load_script(script_file))
    assert docs[0]["character"] == "THREEPIO"
    assert docs[0]["line"] == "We're doomed."


# --- embedding ----------------------------------------------------------------


def test_embed_asks_for_normalized_vectors():
    """The collection is cosine, so the vectors need to be unit length."""
    encoder = FakeEncoder()
    embed(["anything"], encoder)
    assert encoder.calls[0]["normalize"] is True


def test_embed_returns_plain_floats():
    # qdrant-client won't take numpy scalars, and the failure shows up at
    # upsert time as a serialization error rather than anything readable.
    vectors = embed(["the Force"], FakeEncoder())
    assert all(isinstance(x, float) for x in vectors[0])


def test_embed_handles_a_batch():
    assert len(embed(["a", "b", "c"], FakeEncoder())) == 3


# --- creating the collection --------------------------------------------------


def test_create_collection_uses_cosine_and_the_right_width():
    client = FakeClient()
    create_collection(client)
    config = client.created[0]["config"]
    assert config.size == VECTOR_SIZE == 384
    assert config.distance.lower() == "cosine"


def test_create_collection_replaces_an_existing_one():
    """Running the build twice shouldn't leave two copies of every line."""
    client = FakeClient(exists=True)
    create_collection(client, "star-wars")
    assert client.deleted == ["star-wars"]
    assert len(client.created) == 1


def test_create_collection_does_not_delete_what_is_not_there():
    client = FakeClient(exists=False)
    create_collection(client)
    assert client.deleted == []


# --- writing the documents ----------------------------------------------------


def documents(n):
    return [
        {"text": f"line {i}", "character": "X", "line": f"line {i}"} for i in range(n)
    ]


def test_upsert_writes_everything_it_was_given():
    client = FakeClient()
    assert upsert_documents(client, documents(5), FakeEncoder(), batch_size=2) == 5


def test_upsert_batches_rather_than_sending_one_call():
    client = FakeClient()
    upsert_documents(client, documents(5), FakeEncoder(), batch_size=2)
    assert [len(u["points"]) for u in client.upserts] == [2, 2, 1]


def test_upsert_ids_are_unique_across_batches():
    """The bug this is here for would silently lose most of the corpus.

    Numbering points within each batch instead of across the whole run means
    batch two overwrites batch one, and the collection quietly ends up the size
    of a single batch with no error anywhere.
    """
    client = FakeClient()
    upsert_documents(client, documents(7), FakeEncoder(), batch_size=3)
    ids = [p.id for u in client.upserts for p in u["points"]]
    assert ids == list(range(7))
    assert len(set(ids)) == 7


def test_upsert_carries_the_payload_through():
    client = FakeClient()
    upsert_documents(client, documents(1), FakeEncoder())
    assert client.upserts[0]["points"][0].payload["character"] == "X"


def test_upsert_of_nothing_writes_nothing():
    client = FakeClient()
    assert upsert_documents(client, [], FakeEncoder()) == 0
    assert client.upserts == []


# --- searching ----------------------------------------------------------------


def test_search_returns_hits_best_first():
    client = FakeClient(
        hits=[
            FakeHit(0.9, {"text": "BEN: the Force", "character": "BEN"}),
            FakeHit(0.5, {"text": "LUKE: what?", "character": "LUKE"}),
        ]
    )
    results = search(client, "the Force", FakeEncoder())
    assert [r["text"] for r in results] == ["BEN: the Force", "LUKE: what?"]
    assert results[0]["score"] > results[1]["score"]


def test_search_embeds_the_query_and_sends_a_vector():
    client = FakeClient(hits=[FakeHit(0.9, {"text": "x"})])
    search(client, "the Force", FakeEncoder())
    assert isinstance(client.queries[0]["query"], list)
    assert len(client.queries[0]["query"]) == 4


def test_search_respects_the_limit():
    hits = [FakeHit(1.0 - i / 10, {"text": f"doc {i}"}) for i in range(5)]
    assert len(search(FakeClient(hits=hits), "q", FakeEncoder(), limit=2)) == 2


def test_search_keeps_the_payload_fields():
    client = FakeClient(hits=[FakeHit(0.8, {"text": "BEN: hi", "character": "BEN"})])
    assert search(client, "q", FakeEncoder())[0]["character"] == "BEN"


def test_search_survives_a_hit_with_no_text():
    # Qdrant hands back whatever payload is stored, and a document written
    # without a text field shouldn't take the whole query down.
    client = FakeClient(hits=[FakeHit(0.8, {"character": "BEN"})])
    assert search(client, "q", FakeEncoder())[0]["text"] == ""


# --- the whole thing together -------------------------------------------------


def test_build_indexes_every_line_in_the_file(script_file):
    client = FakeClient()
    assert build(client, FakeEncoder(), path=script_file) == 3
    assert client.created[0]["name"] == SCRIPT_COLLECTION


def test_build_then_search_finds_the_right_line(script_file):
    """End to end with fakes, which is the only version CI can run."""
    client = FakeClient(hits=[FakeHit(0.95, {"text": "LUKE: The Force?"})])
    build(client, FakeEncoder(), path=script_file)
    assert search(client, "tell me about the Force", FakeEncoder())[0]["text"] == (
        "LUKE: The Force?"
    )
