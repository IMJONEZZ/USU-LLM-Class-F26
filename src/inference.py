import json
import os
import sys
import time
from itertools import pairwise

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


def completion_request(question, model) -> dict:
    return {
        "model": model,
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
    retries = 0
    while True:
        response = requests.request(
            method, os.environ["MODAL_URL"] + path, headers=headers, **kwargs
        )
        waited = time.perf_counter() - start
        if response.status_code != 503 or waited >= config.RETRY_TIMEOUT_SECONDS:
            break
        retries += 1
        time.sleep(config.RETRY_INTERVAL_SECONDS)
    response.raise_for_status()
    return response, waited, retries


def complete(question, model):
    body = completion_request(question, model)
    return call("POST", "/v1/completions", json=body)[0].json()


def stream_stats(arrivals):
    """arrivals: seconds from sending the request to each streamed token chunk."""
    gaps = [later - earlier for earlier, later in pairwise(arrivals)]
    return {
        "ttft_s": arrivals[0],
        "chunks": len(arrivals),
        "rate_tok_s": (len(arrivals) - 1) / (arrivals[-1] - arrivals[0]),
        "max_gap_s": max(gaps),
    }


def stream_body(question, model):
    body = completion_request(question, model)
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    return body


def parse_sse(lines):
    """Yield each JSON chunk of a server-sent-events stream; blank lines and [DONE] are skipped."""
    for line in lines:
        if line.startswith("data: ") and line != "data: [DONE]":
            yield json.loads(line[len("data: ") :])


def ask(question):
    models, waited, _ = call("GET", "/v1/models")
    print(f"models: {[m['id'] for m in models.json()['data']]} (waited {waited:.0f}s)")
    result = complete(question, config.ADAPTER_NAME)
    choice = result["choices"][0]
    print(choice["text"])
    print(f"finish_reason: {choice['finish_reason']} | usage: {result['usage']}")


def main(command, argument):
    truststore.inject_into_ssl()
    if command == "ask":
        ask(argument)
    else:
        raise ValueError(f"unknown command {command!r}: use ask")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
