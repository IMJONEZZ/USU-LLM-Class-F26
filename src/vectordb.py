import json
import sys
import time
from collections import Counter
from functools import cache
from pathlib import Path

import numpy as np
from google.genai import types
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from src import config, gemini

FIELDS = ["id", "question", "answer"]


def load_questions(path):
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if len(rows) != config.QUESTION_COUNT:
        raise ValueError(
            f"expected {config.QUESTION_COUNT} questions, found {len(rows)}"
        )
    questions = [{field: row[field] for field in FIELDS} for row in rows]
    for q in questions:
        for field in FIELDS:
            if not q[field].strip():
                raise ValueError(f"{q['id']!r}: {field} is empty")
    for question_id, count in Counter(q["id"] for q in questions).items():
        if count > 1:
            raise ValueError(f"duplicate id {question_id!r}")
    return questions


def doc_prompt(question):
    return (
        "Write a factual reference document of about 300 tokens that a reader could "
        "use to answer the question below. It must support the answer. Do not "
        "restate the question and do not mention that this is a document.\n"
        f"Question: {question['question']}\n"
        f"Answer: {question['answer']}"
    )


def validate_docs(questions, docs):
    if set(docs) != {q["id"] for q in questions}:
        raise ValueError("document ids do not match question ids")
    for doc_id, text in docs.items():
        if not config.MIN_DOC_CHARS <= len(text) <= config.MAX_DOC_CHARS:
            raise ValueError(f"{doc_id!r}: document has {len(text)} characters")


def generate_docs(questions, path):
    path = Path(path)
    docs = json.loads(path.read_text()) if path.exists() else {}
    for q in questions:
        if q["id"] not in docs:
            prompt = doc_prompt(q)
            text = gemini._generate(
                prompt, config.TEACHER_MODEL, config.DOC_TEMPERATURE
            )
            docs[q["id"]] = text.strip()
            path.write_text(json.dumps(docs, indent=2))
    validate_docs(questions, docs)
    return docs


@cache
def _embed(text, task_type):
    client = (
        gemini.get_client()
    )  # a temporary client is garbage-collected and closes its connection
    response = client.models.embed_content(
        model=config.EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type, output_dimensionality=config.EMBED_FULL_DIM
        ),
    )
    return tuple(response.embeddings[0].values)


def truncate(vector, dim):
    if dim > config.EMBED_FULL_DIM:
        raise ValueError(f"dim {dim} is above {config.EMBED_FULL_DIM}")
    cut = np.array(vector[:dim], dtype=float)
    norm = np.linalg.norm(cut)
    return (cut / norm if norm else cut).tolist()


class GeminiEmbeddings(Embeddings):
    def __init__(self, dim):
        self.dim = dim

    def embed_documents(self, texts):
        return [truncate(_embed(t, "RETRIEVAL_DOCUMENT"), self.dim) for t in texts]

    def embed_query(self, text):
        return truncate(_embed(text, "RETRIEVAL_QUERY"), self.dim)


def open_store(dim, persist_dir, embeddings=None):
    if dim not in config.DIMS:
        raise ValueError(f"dim {dim} is not one of {config.DIMS}")
    return Chroma(
        collection_name=f"docs_{dim}",
        embedding_function=embeddings or GeminiEmbeddings(dim),
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_store(dim, docs, persist_dir, embeddings=None):
    store = open_store(dim, persist_dir, embeddings)
    if len(store.get()["ids"]) != len(docs):
        store.add_texts(
            list(docs.values()),
            metadatas=[{"id": doc_id} for doc_id in docs],
            ids=list(docs),
        )
    return store


def top1(store, question):
    document, distance = store.similarity_search_with_score(question, k=1)[0]
    return document.metadata["id"], document.page_content, distance


def check_dim(store, questions):
    rows = []
    for q in questions:
        hits = store.similarity_search_with_score(q["question"], k=len(questions))
        distances = {doc.metadata["id"]: d for doc, d in hits}
        winner = hits[0][0].metadata["id"]
        rows.append(
            {
                "id": q["id"],
                "correct": winner == q["id"],
                "winner": winner,
                "own_distance": distances[q["id"]],
                "winner_distance": hits[0][1],
            }
        )
    return rows


def sweep(questions, docs, persist_dir):
    for q in questions:  # embed first so the timings leave out the API calls
        _embed(q["question"], "RETRIEVAL_QUERY")
    results, seconds = {}, {}
    for dim in config.DIMS:
        store = build_store(dim, docs, persist_dir)
        start = time.perf_counter()
        results[dim] = check_dim(store, questions)
        seconds[dim] = time.perf_counter() - start
    return results, seconds


def smallest_perfect(results):
    perfect = [dim for dim, rows in results.items() if all(r["correct"] for r in rows)]
    return min(perfect) if perfect else None


def main(command):
    questions = load_questions(config.TEST_SET_PATH)
    docs = generate_docs(questions, config.DOCS_PATH)
    if command == "docs":
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_pretrained(config.MODEL_NAME)
        for q in questions:
            count = len(tokenizer.encode(docs[q["id"]]).ids)
            print(f"{q['id']}: {count} tokens\n{docs[q['id']]}\n")
    if command == "ask":
        store = build_store(config.EMBED_FULL_DIM, docs, config.CHROMA_DIR)
        doc_id, text, distance = top1(store, questions[0]["question"])
        print(
            f"Q: {questions[0]['question']}\nretrieved {doc_id} (distance {distance:.4f})\n{text}"
        )
    if command == "sweep":
        results, seconds = sweep(questions, docs, config.CHROMA_DIR)
        for dim, rows in results.items():
            missed = [r for r in rows if not r["correct"]]
            print(
                f"{dim:>5}: {len(rows) - len(missed)}/{len(rows)} correct, "
                f"{seconds[dim] * 1000:.1f} ms for {len(rows)} searches"
            )
            for r in missed:
                print(
                    f"   missed {r['id']}: got {r['winner']} "
                    f"({r['winner_distance']:.4f} vs own {r['own_distance']:.4f})"
                )
        print(f"total search time: {sum(seconds.values()) * 1000:.1f} ms")
        print("smallest size with a perfect score:", smallest_perfect(results))


if __name__ == "__main__":
    main(sys.argv[1])
