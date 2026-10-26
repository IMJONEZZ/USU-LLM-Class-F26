"""A vector database over the Star Wars scripts, and the search on top of it.

Qdrant runs embedded here, so `QdrantClient(path=...)` writes to a folder on
disk and there's no server to start and no account to sign up for. The
embeddings come from `BAAI/bge-small-en-v1.5`, which is small enough to run on
CPU and gives 384 dimensions.

The corpus is the same dialogue I've been using since assignment 1, except this
time the speaker comes along with it. Assignment 2 threw the character names
away on purpose, because folding "THREEPIO:" into training text teaches a model
to type it out as if it were speech. Here it's payload rather than training
data, so keeping it costs nothing and makes a result readable.

Both the client and the encoder are arguments. The tests pass fakes, which is
what lets them cover this file without downloading a model or writing anything
to disk, and it makes the ranking tests deterministic because I can hand over
vectors where I already know which one should win.

    uv run python -m src.vectordb "I have a bad feeling about this"
"""

import json

DEFAULT_DATASET = "SW_EpisodeIV_VI.json"
DEFAULT_LOCATION = "qdrant_data"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# bge-small is 384 dimensions. Qdrant wants that up front when the collection
# is created and will reject every vector that doesn't match, so it's a
# constant rather than something inferred at upsert time.
VECTOR_SIZE = 384

SCRIPT_COLLECTION = "star-wars"
BATCH_SIZE = 256


def load_script(path=DEFAULT_DATASET):
    """Read the dialogue, keeping the speaker this time.

    `src.dataloader.load_lines` drops the character on purpose. That was right
    for training and wrong here.
    """
    with open(path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    return [
        {"character": r.get("Character", ""), "line": r["Line"]}
        for r in records
        if isinstance(r, dict) and r.get("Line")
    ]


def to_documents(records):
    """Turn the records into the text that actually gets embedded.

    The speaker goes into the embedded string rather than sitting only in the
    payload, so that asking about a character can match their lines. It costs a
    couple of tokens per document and the alternative is that "THREEPIO" only
    ever matches lines that happen to say the word Threepio.
    """
    return [
        {
            "text": f"{r['character']}: {r['line']}" if r["character"] else r["line"],
            "character": r["character"],
            "line": r["line"],
        }
        for r in records
    ]


def load_encoder(model_name=EMBED_MODEL):  # pragma: no cover - downloads a model
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed(texts, encoder):
    """Embed a list of strings and hand back plain lists of floats.

    Normalized, because the collection is set up for cosine distance and
    normalizing once here means the scores come back on a sane scale.
    """
    vectors = encoder.encode(list(texts), normalize_embeddings=True)
    return [list(map(float, v)) for v in vectors]


def connect(location=DEFAULT_LOCATION):  # pragma: no cover - touches the disk
    from qdrant_client import QdrantClient

    return QdrantClient(path=location)


def create_collection(client, name=SCRIPT_COLLECTION, size=VECTOR_SIZE):
    """Make the collection, replacing it if it's already there.

    Recreating rather than appending means running this twice doesn't leave two
    copies of every line in the index, which is a mistake I'd rather make
    impossible than remember not to make.
    """
    from qdrant_client.models import Distance, VectorParams

    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=size, distance=Distance.COSINE),
    )
    return name


def upsert_documents(
    client, documents, encoder, name=SCRIPT_COLLECTION, batch_size=BATCH_SIZE
):
    """Embed the documents and write them in batches.

    Batched because encoding 2,523 lines one at a time is slow and because
    Qdrant would rather take a few hundred points per call than one.
    """
    from qdrant_client.models import PointStruct

    total = 0
    for start in range(0, len(documents), batch_size):
        chunk = documents[start : start + batch_size]
        vectors = embed([d["text"] for d in chunk], encoder)
        client.upsert(
            collection_name=name,
            points=[
                PointStruct(id=start + i, vector=vector, payload=doc)
                for i, (doc, vector) in enumerate(zip(chunk, vectors))
            ],
        )
        total += len(chunk)
    return total


def search(client, query, encoder, name=SCRIPT_COLLECTION, limit=3):
    """Embed the query and return the closest documents, best first."""
    vector = embed([query], encoder)[0]
    hits = client.query_points(
        collection_name=name, query=vector, limit=limit, with_payload=True
    ).points
    return [
        {"score": hit.score, "text": hit.payload.get("text", ""), **hit.payload}
        for hit in hits
    ]


def build(client, encoder, path=DEFAULT_DATASET, name=SCRIPT_COLLECTION):
    """Load the script, make the collection and fill it. Returns the count."""
    documents = to_documents(load_script(path))
    create_collection(client, name)
    return upsert_documents(client, documents, encoder, name)


if __name__ == "__main__":  # pragma: no cover - entry point, not logic
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else "Who talks about the Force?"

    client, encoder = connect(), load_encoder()
    count = build(client, encoder)
    print(f"indexed {count} lines into '{SCRIPT_COLLECTION}'\n")

    print(f"QUERY  {question}\n")
    for i, hit in enumerate(search(client, question, encoder), 1):
        print(f"  {i}. [{hit['score']:.3f}] {hit['text']}")

    # Closing by hand. Left to the garbage collector this throws an ImportError
    # out of __del__ while the interpreter is shutting down, which looks like a
    # crash at the end of an otherwise fine run.
    client.close()
