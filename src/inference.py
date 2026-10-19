import os
import sys
import time

import requests
import truststore

from src import config

_MARKERS = [config.INSTRUCTION_MARKER.strip(), config.RESPONSE_MARKER.strip()]


def _reject_markers(text, what):
    for marker in _MARKERS:
        if marker in text:
            raise ValueError(f"{what} contains template marker {marker!r}")


def format_prompt(question):
    _reject_markers(question, "question")
    question = question.strip()
    if not question:
        raise ValueError("question is empty")
    return f"{config.INSTRUCTION_MARKER}{question}\n\n{config.RESPONSE_MARKER}"


def check_chain(chain):
    _reject_markers(chain, "chain")
    lines = chain.split("\n")
    positions = []
    for label in config.SECTION_LABELS:
        found = [i for i, line in enumerate(lines) if line.lstrip().startswith(label)]
        if len(found) != 1:
            raise ValueError(f"expected exactly one {label!r} line, found {len(found)}")
        positions.append(found[0])
    if positions != sorted(positions):
        raise ValueError("chain sections are out of order")
    final_line = lines[positions[-1]].lstrip()
    if not final_line[len(config.FINAL_ANSWER_LABEL) :].strip():
        raise ValueError("final answer is empty")


def completion_request(question):
    return {
        "model": config.ADAPTER_NAME,
        "prompt": format_prompt(question),
        "max_tokens": config.EVAL_MAX_NEW_TOKENS,
        "temperature": 0,
    }


def call(method, path, **kwargs):
    """Send one request; a 503 means the server is still starting, so ask again."""
    headers = {
        "Modal-Key": os.environ["MODAL_ID"],
        "Modal-Secret": os.environ["MODAL_SECRET"],
    }
    start = time.perf_counter()
    while True:
        response = requests.request(
            method, os.environ["MODAL_URL"] + path, headers=headers, **kwargs
        )
        waited = time.perf_counter() - start
        if response.status_code != 503 or waited >= config.RETRY_TIMEOUT_SECONDS:
            break
        time.sleep(config.RETRY_INTERVAL_SECONDS)
    response.raise_for_status()
    return response, waited


def main(question):
    truststore.inject_into_ssl()
    models, waited = call("GET", "/v1/models")
    print(f"models: {[m['id'] for m in models.json()['data']]} (waited {waited:.0f}s)")
    body = completion_request(question)
    result = call("POST", "/v1/completions", json=body)[0].json()
    choice = result["choices"][0]
    print(choice["text"])
    print(f"finish_reason: {choice['finish_reason']} | usage: {result['usage']}")


if __name__ == "__main__":
    main(sys.argv[1])
