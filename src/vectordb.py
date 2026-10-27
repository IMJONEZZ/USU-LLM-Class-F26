"""Assignment 8: local ChromaDB retrieval for Assignment 6 Q&A pairs.

Retrieval-only for now: questions are embedded and stored; reference answers
live in metadata. Helpers are structured for later FastAPI / Llama reuse.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api.types import EmbeddingFunction

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "assignment8" / "documents.jsonl"
DEFAULT_PARAPHRASE_PATH = REPO_ROOT / "data" / "assignment8" / "paraphrases.jsonl"
DEFAULT_PERSIST_DIR = REPO_ROOT / "data" / "assignment8" / "chroma"
COLLECTION_NAME = "assignment8_a6_questions"


class QARecord(TypedDict):
    id: str
    question: str
    answer: str
    assumptions: str


class ParaphraseRecord(TypedDict):
    id: str
    expected_id: str
    question: str


class RetrievalHit(TypedDict):
    id: str
    question: str
    answer: str
    distance: float | None


class RetrievalScore(TypedDict):
    correct: int
    total: int
    success_rate: float
    details: list[dict[str, Any]]


def load_documents(path: Path | str = DEFAULT_DATA_PATH) -> list[QARecord]:
    """Load Assignment 6 Q&A records from JSONL (id, question, answer)."""
    records: list[QARecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            records.append(
                {
                    "id": str(raw["id"]),
                    "question": str(raw["question"]),
                    "answer": str(raw["answer"]),
                    "assumptions": str(raw.get("assumptions", "")),
                }
            )
    return records


def load_paraphrases(
    path: Path | str = DEFAULT_PARAPHRASE_PATH,
) -> list[ParaphraseRecord]:
    """Load optional paraphrase probes used as an extra retrieval check."""
    records: list[ParaphraseRecord] = []
    paraphrase_path = Path(path)
    if not paraphrase_path.is_file():
        return records
    with paraphrase_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            records.append(
                {
                    "id": str(raw["id"]),
                    "expected_id": str(raw["expected_id"]),
                    "question": str(raw["question"]),
                }
            )
    return records


def create_persistent_client(
    persist_directory: Path | str = DEFAULT_PERSIST_DIR,
) -> chromadb.ClientAPI:
    """Create a ChromaDB client that stores data under persist_directory."""
    path = Path(persist_directory)
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def get_or_create_docs_collection(
    client: chromadb.ClientAPI,
    name: str = COLLECTION_NAME,
    embedding_function: EmbeddingFunction | None = None,
) -> Collection:
    """Get or create the Q&A collection.

    When embedding_function is omitted, Chroma uses its default local MiniLM.
    Tests pass a fixed embedding function so CI stays offline.
    """
    if embedding_function is None:
        return client.get_or_create_collection(name=name)
    return client.get_or_create_collection(
        name=name, embedding_function=embedding_function
    )


def upsert_documents(
    collection: Collection,
    documents: list[QARecord],
    embeddings: list[list[float]] | None = None,
) -> list[str]:
    """Upsert Q&A records: embed questions; store answers in metadata."""
    ids = [doc["id"] for doc in documents]
    questions = [doc["question"] for doc in documents]
    metadatas: list[dict[str, str]] = [
        {
            "answer": doc["answer"],
            "assumptions": doc.get("assumptions", ""),
        }
        for doc in documents
    ]
    kwargs: dict[str, Any] = {
        "ids": ids,
        "documents": questions,
        "metadatas": metadatas,
    }
    if embeddings is not None:
        kwargs["embeddings"] = embeddings
    collection.upsert(**kwargs)
    return ids


def query_similar(
    collection: Collection,
    query_texts: list[str] | None = None,
    query_embeddings: list[list[float]] | None = None,
    n_results: int = 1,
) -> dict[str, Any]:
    """Run a similarity search with either query texts or fixed embeddings."""
    if query_texts is None and query_embeddings is None:
        raise ValueError("Provide query_texts or query_embeddings")
    kwargs: dict[str, Any] = {
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if query_texts is not None:
        kwargs["query_texts"] = query_texts
    if query_embeddings is not None:
        kwargs["query_embeddings"] = query_embeddings
    return collection.query(**kwargs)


def _hits_from_query_results(results: dict[str, Any]) -> list[list[RetrievalHit]]:
    """Convert a Chroma query payload into structured retrieval hits."""
    all_hits: list[list[RetrievalHit]] = []
    ids = results.get("ids") or []
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    distances = results.get("distances") or []
    for index in range(len(ids)):
        hits: list[RetrievalHit] = []
        hit_ids = ids[index] if index < len(ids) else []
        hit_docs = documents[index] if index < len(documents) else []
        hit_meta = metadatas[index] if index < len(metadatas) else []
        hit_distances = distances[index] if index < len(distances) else []
        for rank, doc_id in enumerate(hit_ids):
            meta = hit_meta[rank] if rank < len(hit_meta) and hit_meta[rank] else {}
            hits.append(
                {
                    "id": doc_id,
                    "question": hit_docs[rank] if rank < len(hit_docs) else "",
                    "answer": str(meta.get("answer", "")),
                    "distance": hit_distances[rank]
                    if rank < len(hit_distances)
                    else None,
                }
            )
        all_hits.append(hits)
    return all_hits


def retrieve_answers(
    collection: Collection,
    questions: list[str],
    n_results: int = 1,
    query_embeddings: list[list[float]] | None = None,
) -> list[list[RetrievalHit]]:
    """Retrieve top matching reference answers for each question.

    Intended for reuse by a later FastAPI / Llama RAG step.
    """
    if query_embeddings is not None:
        results = query_similar(
            collection,
            query_embeddings=query_embeddings,
            n_results=n_results,
        )
    else:
        results = query_similar(
            collection,
            query_texts=questions,
            n_results=n_results,
        )
    return _hits_from_query_results(results)


def format_retrieval_report(
    questions: list[str],
    hits_per_query: list[list[RetrievalHit]],
) -> str:
    """Pretty-print queries with retrieved answer, distance, and record ID."""
    lines: list[str] = []
    for index, question in enumerate(questions):
        lines.append(f"Query: {question}")
        hits = hits_per_query[index] if index < len(hits_per_query) else []
        if not hits:
            lines.append("  (no results)")
            continue
        for rank, hit in enumerate(hits, start=1):
            distance = hit["distance"]
            distance_text = f", distance={distance:.4f}" if distance is not None else ""
            lines.append(
                f"  {rank}. id={hit['id']}{distance_text}\n     answer: {hit['answer']}"
            )
    return "\n".join(lines)


def format_query_results(query_texts: list[str], results: dict[str, Any]) -> str:
    """Compatibility wrapper around structured retrieval formatting."""
    return format_retrieval_report(query_texts, _hits_from_query_results(results))


def score_retrieval(
    questions: list[str],
    expected_ids: list[str],
    hits_per_query: list[list[RetrievalHit]],
) -> RetrievalScore:
    """Score whether each query's top hit matches the expected record ID."""
    details: list[dict[str, Any]] = []
    correct = 0
    total = len(questions)
    for index, question in enumerate(questions):
        expected = expected_ids[index]
        hits = hits_per_query[index] if index < len(hits_per_query) else []
        top = hits[0] if hits else None
        matched = top is not None and top["id"] == expected
        if matched:
            correct += 1
        details.append(
            {
                "question": question,
                "expected_id": expected,
                "retrieved_id": None if top is None else top["id"],
                "distance": None if top is None else top["distance"],
                "answer": "" if top is None else top["answer"],
                "matched": matched,
            }
        )
    rate = (correct / total) if total else 0.0
    return {
        "correct": correct,
        "total": total,
        "success_rate": rate,
        "details": details,
    }


def evaluate_exact_questions(
    collection: Collection,
    records: list[QARecord],
    query_embeddings: list[list[float]] | None = None,
) -> RetrievalScore:
    """Benchmark: query with the original 10 Assignment 6 questions."""
    questions = [record["question"] for record in records]
    expected_ids = [record["id"] for record in records]
    hits = retrieve_answers(
        collection,
        questions,
        n_results=1,
        query_embeddings=query_embeddings,
    )
    return score_retrieval(questions, expected_ids, hits)


def evaluate_paraphrases(
    collection: Collection,
    paraphrases: list[ParaphraseRecord],
    query_embeddings: list[list[float]] | None = None,
) -> RetrievalScore:
    """Extra check: paraphrased questions mapped to expected record IDs."""
    questions = [record["question"] for record in paraphrases]
    expected_ids = [record["expected_id"] for record in paraphrases]
    hits = retrieve_answers(
        collection,
        questions,
        n_results=1,
        query_embeddings=query_embeddings,
    )
    return score_retrieval(questions, expected_ids, hits)


def build_collection(
    persist_directory: Path | str = DEFAULT_PERSIST_DIR,
    data_path: Path | str = DEFAULT_DATA_PATH,
    documents: list[QARecord] | None = None,
    embedding_function: EmbeddingFunction | None = None,
    embeddings: list[list[float]] | None = None,
) -> tuple[chromadb.ClientAPI, Collection, list[QARecord]]:
    """Create the persistent collection and upsert Q&A records.

    Pass ``embeddings`` to skip Chroma's default local embedding model (useful
    for offline tests). Omit it in production to use MiniLM. ``documents`` may
    be supplied directly; otherwise records are loaded from ``data_path``.
    """
    records = list(documents) if documents is not None else load_documents(data_path)
    client = create_persistent_client(persist_directory)
    collection = get_or_create_docs_collection(
        client, embedding_function=embedding_function
    )
    upsert_documents(collection, records, embeddings=embeddings)
    return client, collection, records


def main(
    persist_directory: Path | str = DEFAULT_PERSIST_DIR,
    data_path: Path | str = DEFAULT_DATA_PATH,
    paraphrase_path: Path | str = DEFAULT_PARAPHRASE_PATH,
    documents: list[QARecord] | None = None,
    queries: list[str] | None = None,
    embedding_function: EmbeddingFunction | None = None,
    embeddings: list[list[float]] | None = None,
    query_embeddings: list[list[float]] | None = None,
    n_results: int = 1,
    run_paraphrases: bool = True,
) -> int:
    """Initialize the DB, run the exact benchmark, optionally paraphrases."""
    _client, collection, documents = build_collection(
        persist_directory=persist_directory,
        data_path=data_path,
        documents=documents,
        embedding_function=embedding_function,
        embeddings=embeddings,
    )
    print(f"Stored documents: {collection.count()}")
    print(f"Dataset size: {len(documents)}")

    if queries is not None:
        hits = retrieve_answers(
            collection,
            queries,
            n_results=n_results,
            query_embeddings=query_embeddings,
        )
        print(format_retrieval_report(queries, hits))
        return 0

    print("\n=== Exact Assignment 6 questions (benchmark) ===")
    exact_questions = [record["question"] for record in documents]
    exact_ids = [record["id"] for record in documents]
    exact_hits = retrieve_answers(
        collection,
        exact_questions,
        n_results=n_results,
        query_embeddings=query_embeddings,
    )
    print(format_retrieval_report(exact_questions, exact_hits))
    exact_score = score_retrieval(exact_questions, exact_ids, exact_hits)
    print(
        f"\nExact retrieval success: {exact_score['correct']}/"
        f"{exact_score['total']} "
        f"({exact_score['success_rate']:.0%})"
    )

    if run_paraphrases:
        paraphrases = load_paraphrases(paraphrase_path)
        if paraphrases:
            print("\n=== Paraphrased questions (extra check) ===")
            para_questions = [record["question"] for record in paraphrases]
            para_expected = [record["expected_id"] for record in paraphrases]
            para_hits = retrieve_answers(
                collection, para_questions, n_results=n_results
            )
            print(format_retrieval_report(para_questions, para_hits))
            para_score = score_retrieval(para_questions, para_expected, para_hits)
            print(
                f"\nParaphrase retrieval success: {para_score['correct']}/"
                f"{para_score['total']} "
                f"({para_score['success_rate']:.0%})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
