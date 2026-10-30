import json
import os
import sys
import time
from pathlib import Path

import requests
import truststore

from src import config, gemini, vectordb


def build_prompt(document, question):
    return f"{document}\n\nQuestion: {question}\nAnswer:"


def complete(prompt):
    start = time.monotonic()
    while True:
        attempt = time.monotonic()
        response = requests.post(
            os.environ["MODAL_URL"] + "/v1/completions",
            headers={
                "Modal-Key": os.environ["MODAL_ID"],
                "Modal-Secret": os.environ["MODAL_SECRET"],
            },
            json={
                "model": config.SERVED_BASE_NAME,
                "prompt": prompt,
                "max_tokens": config.ANSWER_MAX_TOKENS,
                "temperature": 0,
                "stop": config.ANSWER_STOP,
            },
        )
        if (
            response.status_code != 503
            or attempt - start >= config.RETRY_TIMEOUT_SECONDS
        ):
            break
        time.sleep(config.RETRY_INTERVAL_SECONDS)
    response.raise_for_status()
    # the second value is the time spent on 503s before the request that worked
    return response.json()["choices"][0]["text"].strip(), attempt - start


def judge_prompt(question, answer, response):
    return (
        "Grade one response to a question using only the answer key. Do not use "
        "your own knowledge of the world.\n"
        f"Question: {question}\n"
        f"Answer key: {answer}\n"
        f"Response to grade: {response}\n"
        "correct: true only if the response gives the same answer as the key, even "
        "if it is worded differently or has extra words; false otherwise.\n"
        'Reply with JSON only, like {"correct": true}.'
    )


def parse_verdict(text):
    return '"correct":true' in text.lower().replace(
        " ", ""
    )  # anything unparseable is false


def judge_answer(question, answer, response):
    prompt = judge_prompt(question, answer, response)
    text = gemini._generate(prompt, config.JUDGE_MODEL, config.JUDGE_TEMPERATURE)
    return parse_verdict(text)


def score(results):
    return sum(r["answer_correct"] for r in results)


def mean_seconds(results):
    return sum(r["seconds"] for r in results) / len(results)


def run_all(questions, store):
    results = []
    for q in questions:
        start = (
            time.perf_counter()
        )  # a user's wait: embed, search, answer; no 503s, no judge
        doc_id, document, _ = vectordb.top1(store, q["question"])
        model_answer, retry_wait = complete(build_prompt(document, q["question"]))
        seconds = round(time.perf_counter() - start - retry_wait, 2)
        results.append(
            {
                "seconds": seconds,
                "id": q["id"],
                "retrieved_id": doc_id,
                "retrieval_correct": doc_id == q["id"],
                "key": q["answer"],
                "model_answer": model_answer,
                "answer_correct": judge_answer(
                    q["question"], q["answer"], model_answer
                ),
            }
        )
    return results


def main(dim):
    truststore.inject_into_ssl()
    questions = vectordb.load_questions(config.TEST_SET_PATH)
    docs = vectordb.generate_docs(questions, config.DOCS_PATH)
    store = vectordb.build_store(dim, docs, config.CHROMA_DIR)
    results = run_all(questions, store)
    Path(config.RESULTS_PATH).parent.mkdir(exist_ok=True)
    Path(config.RESULTS_PATH).write_text(json.dumps(results, indent=2))
    for r in results:
        print(
            f"{r['id'][:28]:<28} retrieval={r['retrieval_correct']!s:<5} "
            f"answer={r['answer_correct']!s:<5} key={r['key']!r} got={r['model_answer']!r}"
        )
    print(f"correct: {score(results)}/{len(results)}")
    print(f"average time to a completed response: {mean_seconds(results):.2f} s")


if __name__ == "__main__":
    main(int(sys.argv[1]))
