"""A vector database over the Star Wars scripts, and the search on top of it.

Qdrant runs embedded here, so `QdrantClient(path=...)` writes to a folder on
disk and there's no server to start and no account to sign up for. The
embeddings come from `BAAI/bge-small-en-v1.5`, which is small enough to run on
CPU and gives 384 dimensions.

The corpus is the same dialogue I've been using since assignment 1, except this
time the speaker comes along with it. Assignment 2 threw the character names
away on purpose, because folding "THREEPIO:" into training text teaches a model
to type it out as if it were speech. Here it's payload, not training
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
# constant instead of something worked out at upsert time.
VECTOR_SIZE = 384

SCRIPT_COLLECTION = "star-wars"
QA_COLLECTION = "trick-questions"
BATCH_SIZE = 256

# The ten questions from assignment 6 with the answers they should have had.
# The extra credit is to see whether handing these to the base model through
# retrieval gets them right, so these are the documents that go in the second
# collection. I wrote them, which means a wrong one here turns into a
# confidently wrong answer out the other end, so the two I was least sure
# about I went and checked: knuckle cracking has no demonstrated link to
# arthritis, and Talos was made by Hephaestus, not the Cyclopes that
# the model kept insisting on.
QA_CORPUS = [
    (
        "If it takes 1 hour for 60 people to play an Opera, how many hours will it take 600 people to play the same opera?",
        "It still takes 1 hour. An opera runs as long as it runs, and adding more performers doesn't make the piece shorter.",
    ),
    (
        "Is a pound of feathers or a British pound heavier?",
        "A pound of feathers. A pound of feathers is a unit of weight, while a British pound is money and has no weight of its own.",
    ),
    (
        "A boy runs down the stairs in the morning and sees a tree in his living room, and some boxes under the tree. What's going on?",
        "It's Christmas morning. A decorated tree indoors with wrapped boxes underneath it is a Christmas tree and presents.",
    ),
    (
        "What happens if you crack your knuckles a lot?",
        "Nothing much happens. Cracking your knuckles has not been shown to cause arthritis, and studies comparing habitual crackers with non-crackers find the same rates of hand osteoarthritis.",
    ),
    (
        "If there is a shark in the pool of my basement, is it safe to go upstairs?",
        "Yes, it's safe. A shark can't survive out of water or climb a staircase, so it can't follow you upstairs.",
    ),
    (
        "How much wood could a wood chuck chuck if there were only 5 pounds of wood in the world?",
        "At most 5 pounds, because that is all the wood there is. It cannot chuck more wood than exists.",
    ),
    (
        "Who is the current President of the United States?",
        "Donald Trump is the current President of the United States. He was inaugurated for a second term in January 2025.",
    ),
    (
        "Was Talos alive?",
        "Not in the biological sense. Talos was a giant bronze automaton built by Hephaestus to guard Crete. He moved and acted, but he was a made object rather than a living creature, and Medea destroyed him by releasing the ichor from the single vein in his ankle.",
    ),
    (
        "How many Ls are in the word parallel?",
        "There are 3. Spelling it out, p-a-r-a-l-l-e-l, the Ls are the 5th, 6th and 8th letters.",
    ),
    (
        "What is the riddle of the sphinx, and what are all possible answers satisfying all conditions?",
        "The riddle is: what walks on four legs in the morning, two legs at noon, and three legs in the evening? The answer is a human being, who crawls on all fours as a baby, walks upright as an adult, and leans on a cane in old age.",
    ),
]


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


def qa_documents(corpus=None):
    """Turn the question and answer pairs into documents.

    Only the question gets embedded, not the answer. The thing being matched
    against is somebody asking a question, so burying it in a paragraph of
    answer text just dilutes the part that has to match.
    """
    corpus = QA_CORPUS if corpus is None else corpus
    return [
        {"text": question, "question": question, "answer": answer}
        for question, answer in corpus
    ]


def build_qa(client, encoder, corpus=None, name=QA_COLLECTION):
    """Load the trick questions into their own collection."""
    create_collection(client, name)
    return upsert_documents(client, qa_documents(corpus), encoder, name)


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
