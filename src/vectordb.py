"""Persistent vector database of SQuAD paragraphs, built on Chroma.

Flow: download SQuAD -> keep unique paragraphs -> embed them -> store them in a
Chroma collection on disk -> query with a question -> measure hit@k and latency.
The embedding function is passed in as a parameter so tests can use a fake one.
"""

import argparse
import json
import statistics
import time
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
COLLECTION_NAME = "squad_paragraphs"
KS = (1, 3, 5)
# Pass thresholds written in the Action Plan before any run.
HIT_THRESHOLDS = {1: 0.60, 3: 0.75, 5: 0.85}
LATENCY_THRESHOLD_S = 0.25
# Up to this share of paragraphs may exceed the embedder's token limit and be
# kept (and reported); above it the long ones are skipped.
MAX_OVER_LIMIT_SHARE = 0.05

Embed = Callable[[list[str]], list[list[float]]]
CountTokens = Callable[[str], int]


def download_squad(path: Path, url: str = SQUAD_URL) -> Path:
    """Download the SQuAD file once and keep the local copy."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())
    return path


def load_squad(path: Path) -> list[dict]:
    """Return unique paragraphs in file order, each with its first question.

    SQuAD repeats a paragraph once per group of questions, so duplicates are
    removed by paragraph text. Paragraphs with no questions are dropped.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    seen: set[str] = set()
    items: list[dict] = []
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            if context in seen or not paragraph["qas"]:
                continue
            seen.add(context)
            items.append(
                {
                    "id": f"p{len(items):04d}",
                    "context": context,
                    "question": paragraph["qas"][0]["question"],
                }
            )
    return items


def select_paragraphs(
    items: Sequence[dict],
    n: int,
    limit: int,
    count_tokens: CountTokens,
    max_share: float = MAX_OVER_LIMIT_SHARE,
) -> tuple[list[dict], dict]:
    """Pick n paragraphs, applying the plan's rule for over-long paragraphs.

    If at most max_share of the first n exceed the token limit, keep them all
    and report the count. Otherwise skip the long ones and take the next
    paragraphs in file order until there are n. Nothing is dropped silently:
    the report says how many were over the limit and how many were skipped.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    first = list(items[:n])
    over = sum(1 for item in first if count_tokens(item["context"]) > limit)
    report = {"limit": limit, "checked": len(first), "over_limit": over, "skipped": 0}
    if over <= max_share * len(first):
        return first, report

    kept: list[dict] = []
    skipped = 0
    for item in items:
        if count_tokens(item["context"]) > limit:
            skipped += 1
            continue
        kept.append(item)
        if len(kept) == n:
            break
    report["skipped"] = skipped
    return kept, report


def build_collection(
    client: chromadb.ClientAPI,
    items: Sequence[dict],
    embed: Embed,
    name: str = COLLECTION_NAME,
) -> tuple[chromadb.Collection, int]:
    """Create (or reopen) the cosine collection and add paragraphs not yet in it.

    Chroma defaults to L2 distance, so cosine is set explicitly. Paragraphs
    already stored are not embedded again, which is what makes the on-disk
    database worth keeping. Returns the collection and how many were added.
    """
    if not items:
        raise ValueError("no paragraphs to index")
    collection = client.get_or_create_collection(
        name,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )
    ids = [item["id"] for item in items]
    stored = set(collection.get(ids=ids)["ids"])
    new = [item for item in items if item["id"] not in stored]
    if new:
        documents = [item["context"] for item in new]
        collection.add(
            ids=[item["id"] for item in new],
            documents=documents,
            embeddings=embed(documents),
            metadatas=[{"question": item["question"]} for item in new],
        )
    return collection, len(new)


def query(
    collection: chromadb.Collection, text: str, embed: Embed, k: int = 3
) -> list[dict]:
    """Return the k closest paragraphs as dicts with id, document and distance."""
    if not text.strip():
        raise ValueError("query text is empty")
    if k < 1:
        raise ValueError("k must be at least 1")
    count = collection.count()
    if count == 0:
        return []
    result = collection.query(query_embeddings=embed([text]), n_results=min(k, count))
    return [
        {"id": doc_id, "document": document, "distance": distance}
        for doc_id, document, distance in zip(
            result["ids"][0], result["documents"][0], result["distances"][0]
        )
    ]


def evaluate(
    collection: chromadb.Collection,
    items: Sequence[dict],
    embed: Embed,
    ks: Sequence[int] = KS,
) -> dict:
    """Ask every item's question once and measure hit@k and latency.

    One query with the largest k answers all the smaller k values, because the
    top-1 is the first entry of the top-5. A hit means the source paragraph is
    among the top k.
    """
    if not items:
        raise ValueError("no items to evaluate")
    largest = max(ks)
    hits = {k: 0 for k in ks}
    latencies: list[float] = []
    for item in items:
        start = time.perf_counter()
        results = query(collection, item["question"], embed, largest)
        latencies.append(time.perf_counter() - start)
        ranked = [result["id"] for result in results]
        for k in ks:
            if item["id"] in ranked[:k]:
                hits[k] += 1
    return {
        "queries": len(items),
        "hit_rates": {k: hits[k] / len(items) for k in ks},
        "median_latency_s": statistics.median(latencies),
        "first_latency_s": latencies[0],
        "max_latency_s": max(latencies),
    }


def default_embedder() -> tuple[Embed, CountTokens, int]:
    """Chroma's default MiniLM embedder plus a token counter and its token limit.

    The token counter uses a fresh tokenizer with truncation off, so it counts
    the real length. The limit is read from the embedder's own tokenizer.
    """
    model = ONNXMiniLM_L6_V2()

    def embed(texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vector] for vector in model(texts)]

    embed(["warm up"])  # downloads the model on first use
    limit = model.tokenizer.truncation["max_length"]
    tokenizer = model.Tokenizer.from_file(
        str(model.DOWNLOAD_PATH / model.EXTRACTED_FOLDER_NAME / "tokenizer.json")
    )
    tokenizer.no_truncation()
    tokenizer.no_padding()

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text).ids)

    return embed, count_tokens, limit


def format_report(
    selection: dict, added: int, total: int, metrics: dict, examples: list[str]
) -> str:
    """Build the text shown in the screenshot of a real run."""
    lines = [
        f"Paragraphs: {total} in the database ({added} newly embedded)",
        "Token limit {limit}: {over_limit} of {checked} paragraphs over the limit, "
        "{skipped} skipped".format(**selection),
        "",
        *examples,
        "",
        f"Queries: {metrics['queries']}",
    ]
    for k, rate in metrics["hit_rates"].items():
        verdict = "PASS" if rate >= HIT_THRESHOLDS.get(k, 0) else "MISS"
        lines.append(
            f"  hit@{k}: {rate:.1%} (target {HIT_THRESHOLDS.get(k, 0):.0%}) {verdict}"
        )
    median = metrics["median_latency_s"]
    verdict = "PASS" if median < LATENCY_THRESHOLD_S else "MISS"
    lines.append(
        f"  median query latency: {median * 1000:.0f} ms "
        f"(target under {LATENCY_THRESHOLD_S * 1000:.0f} ms) {verdict}"
    )
    lines.append(
        f"  first query: {metrics['first_latency_s'] * 1000:.0f} ms, "
        f"slowest: {metrics['max_latency_s'] * 1000:.0f} ms"
    )
    return "\n".join(lines)


def describe_example(collection, item: dict, embed: Embed, k: int = 3) -> str:
    """Show one question, the expected paragraph and what the database returned."""
    results = query(collection, item["question"], embed, k)
    found = [result["id"] for result in results]
    mark = "HIT" if item["id"] in found else "MISS"
    top = ", ".join(f"{r['id']} ({r['distance']:.3f})" for r in results)
    return f"Q: {item['question']}\n  expected {item['id']} | top {k}: {top} | {mark}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--db-path", type=Path, default=Path("chroma_db"))
    parser.add_argument("--n", type=int, default=200, help="paragraphs to index")
    args = parser.parse_args(argv)

    start = time.perf_counter()
    embed, count_tokens, limit = default_embedder()
    print(f"Embedder ready in {time.perf_counter() - start:.1f} s")

    squad_path = download_squad(args.data_dir / "dev-v1.1.json")
    items = load_squad(squad_path)
    selected, selection = select_paragraphs(items, args.n, limit, count_tokens)

    client = chromadb.PersistentClient(path=str(args.db_path))
    collection, added = build_collection(client, selected, embed)
    examples = [describe_example(collection, item, embed) for item in selected[:3]]
    metrics = evaluate(collection, selected, embed)
    print(format_report(selection, added, collection.count(), metrics, examples))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
