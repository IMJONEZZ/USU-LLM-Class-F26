"""Extra credit: answer the ten Assignment 6 questions with and without retrieval.

Flow: put the 200 SQuAD paragraphs and the ten question-and-answer notes in a
Chroma collection -> for each question fetch the top k notes -> pass them to the
base Llama (served by Ollama) in one fixed, neutral prompt -> save the answers.
No prompt tuning, no fine-tuning, no adapter: only the base model.
"""

import argparse
import json
import statistics
import time
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

import chromadb

from src import vectordb

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:1b"
RAG_COLLECTION = "rag_knowledge"
# Written before any result was seen (Decision 8). Nothing is added to it.
PROMPT_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}\nAnswer:"

# The ten Assignment 6 questions with the "must include" answers from my own
# correctness guide (A6_CORRECTNESS_GUIDE.md), worded as short notes.
QUESTIONS = [
    (
        (
            "If it takes 1 hour for 60 people to play an Opera, how many "
            "hours will it take 600 people to play the same opera?"
        ),
        (
            "1 hour. An opera takes the same amount of time to perform "
            "whether 60 or 600 people perform it."
        ),
    ),
    (
        "Is a pound of feathers or a British pound heavier?",
        (
            "A pound of feathers is heavier. It weighs 1 pound, a unit of "
            "weight, while a British pound is a unit of currency, not weight."
        ),
    ),
    (
        (
            "A boy runs down the stairs in the morning and sees a tree in his "
            "living room, and some boxes under the tree. What's going on?"
        ),
        (
            "It is Christmas morning (or the Christmas season). The tree is a "
            "Christmas tree and the boxes underneath are wrapped presents."
        ),
    ),
    (
        "What happens if you crack your knuckles a lot?",
        (
            "Cracking your knuckles frequently does not cause arthritis, "
            "according to available scientific evidence. The popping sound "
            "comes from gas cavity formation in the joint fluid, and habitual "
            "knuckle cracking is generally harmless."
        ),
    ),
    (
        ("If there is a shark in the pool of my basement, is it safe to go upstairs?"),
        (
            "Yes, it is generally safe to go upstairs if the shark is "
            "confined to the basement pool. Sharks cannot leave the water and "
            "climb stairs."
        ),
    ),
    (
        (
            "How much wood could a wood chuck chuck if there were only 5 "
            "pounds of wood in the world?"
        ),
        (
            "At most 5 pounds of wood, because that is all the wood available "
            "in the world. A woodchuck cannot chuck more wood than exists."
        ),
    ),
    (
        "Who is the current President of the United States?",
        (
            "Donald Trump is the current President of the United States (as "
            "of October 13, 2026)."
        ),
    ),
    (
        "Was Talos alive?",
        (
            "In Greek mythology, Talos was a giant bronze automaton that "
            "guarded Crete. He was not biologically alive like a human, but "
            "was animated and capable of movement and action. Whether he was "
            "alive depends on how life is defined."
        ),
    ),
    (
        "How many Ls are in the word parallel?",
        "There are 3 Ls in the word parallel.",
    ),
    (
        (
            "What is the riddle of the sphinx, and what are all possible "
            "answers satisfying all conditions?"
        ),
        (
            "The riddle is: What walks on four legs in the morning, two legs "
            "at noon, and three legs in the evening? The traditional answer "
            "is a human being: a baby crawls on all fours, an adult walks on "
            "two legs, and an elderly person may use a walking stick as a "
            "third leg. There is no definitive exhaustive list of all "
            "possible answers."
        ),
    ),
]

Llm = Callable[[str], str]


def qa_items() -> list[dict]:
    """The ten notes in the shape build_collection expects."""
    return [
        {
            "id": f"qa{number:02d}",
            "context": f"Question: {question}\nAnswer: {answer}",
            "question": question,
        }
        for number, (question, answer) in enumerate(QUESTIONS, start=1)
    ]


def build_prompt(question: str, passages: Sequence[str]) -> str:
    """The one fixed prompt: the retrieved text, then the question."""
    return PROMPT_TEMPLATE.format(context="\n\n".join(passages), question=question)


def ask_ollama(prompt: str, model: str = MODEL, url: str = OLLAMA_URL) -> str:
    """Send one prompt to Ollama with greedy decoding and a 512-token cap.

    These are the settings used in Assignments 6 and 7: no system prompt,
    no sampling.
    """
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 512},
        }
    ).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())["message"]["content"]


def answer_question(
    question: str,
    llm: Llm,
    collection: chromadb.Collection | None = None,
    embed: vectordb.Embed | None = None,
    k: int = 3,
) -> dict:
    """Answer one question. With a collection, retrieve first; without, ask directly."""
    retrieved: list[dict] = []
    prompt = question
    if collection is not None:
        if embed is None:
            raise ValueError("retrieval needs an embedding function")
        retrieved = vectordb.query(collection, question, embed, k)
        prompt = build_prompt(question, [hit["document"] for hit in retrieved])
    start = time.perf_counter()
    text = llm(prompt)
    return {
        "question": question,
        "prompt": prompt,
        "retrieved_ids": [hit["id"] for hit in retrieved],
        "answer": text,
        "seconds": time.perf_counter() - start,
    }


def run_all(
    llm: Llm,
    collection: chromadb.Collection,
    embed: vectordb.Embed,
    k: int = 3,
) -> dict:
    """Ask every question without and with retrieval, after one throwaway call.

    The throwaway call keeps model load time out of the answer times, which is
    how "warm" is defined in the plan.
    """
    llm("Hello")
    rows = []
    for number, (question, _) in enumerate(QUESTIONS, start=1):
        plain = answer_question(question, llm)
        rag = answer_question(question, llm, collection, embed, k)
        rows.append(
            {
                "number": number,
                "question": question,
                "without_retrieval": plain,
                "with_retrieval": rag,
                "own_note_retrieved": f"qa{number:02d}" in rag["retrieved_ids"],
            }
        )
    return {
        "k": k,
        "median_seconds_without": statistics.median(
            r["without_retrieval"]["seconds"] for r in rows
        ),
        "median_seconds_with": statistics.median(
            r["with_retrieval"]["seconds"] for r in rows
        ),
        "own_notes_retrieved": sum(r["own_note_retrieved"] for r in rows),
        "rows": rows,
    }


def format_results(results: dict) -> str:
    """Plain-text summary for the screenshot; my verdicts are added by hand."""
    lines = [
        (
            f"k = {results['k']} | own note in the top k: "
            f"{results['own_notes_retrieved']} of {len(results['rows'])} questions"
        ),
        (
            f"median answer time: {results['median_seconds_without']:.1f} s without "
            f"retrieval, {results['median_seconds_with']:.1f} s with retrieval"
        ),
        "",
    ]
    for row in results["rows"]:
        lines.append(f"Q{row['number']}. {row['question']}")
        lines.append(f"  note retrieved: {row['own_note_retrieved']}")
        lines.append(f"  WITHOUT: {row['without_retrieval']['answer']!r}")
        lines.append(f"  WITH:    {row['with_retrieval']['answer']!r}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--db-path", type=Path, default=Path("chroma_db"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n", type=int, default=200, help="SQuAD paragraphs to index")
    parser.add_argument("--out", type=Path, default=Path("rag_results.json"))
    args = parser.parse_args(argv)

    embed, count_tokens, limit = vectordb.default_embedder()
    squad = vectordb.load_squad(
        vectordb.download_squad(args.data_dir / "dev-v1.1.json")
    )
    paragraphs, _ = vectordb.select_paragraphs(squad, args.n, limit, count_tokens)
    client = chromadb.PersistentClient(path=str(args.db_path))
    collection, _ = vectordb.build_collection(
        client, paragraphs + qa_items(), embed, RAG_COLLECTION
    )

    results = run_all(
        lambda prompt: ask_ollama(prompt, args.model), collection, embed, args.k
    )
    results["model"] = args.model
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(format_results(results))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
