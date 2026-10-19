import json
import os
import sys
import time
from pathlib import Path

import requests
import truststore
from tokenizers import Tokenizer

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


def completion_request(question, model):
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


def complete(question, model):
    body = completion_request(question, model)
    return call("POST", "/v1/completions", json=body)[0].json()


def local_token_count(tokenizer, question):
    return len(tokenizer.encode(format_prompt(question), add_special_tokens=True).ids)


def is_format_ok(reply):
    try:
        check_chain(reply.strip())
    except ValueError:
        return False
    return True


def check(path, tokenizer):
    questions = [json.loads(line) for line in Path(path).read_text().splitlines()]
    records = []
    for model in (config.ADAPTER_NAME, config.SERVED_BASE_NAME):
        for row in questions:
            result = complete(row["question"], model)
            choice = result["choices"][0]
            records.append(
                {
                    "id": row["id"],
                    "model": model,
                    "format_ok": is_format_ok(choice["text"]),
                    "finish_reason": choice["finish_reason"],
                    "prompt_tokens": result["usage"]["prompt_tokens"],
                    "local_tokens": local_token_count(tokenizer, row["question"]),
                    "reply": choice["text"],
                }
            )
    Path(config.RUNS_DIR).mkdir(exist_ok=True)
    out = Path(config.RUNS_DIR) / "m4_correctness.json"
    out.write_text(json.dumps(records, indent=2))
    return records


def print_summary(records):
    for model in dict.fromkeys(r["model"] for r in records):
        mine = [r for r in records if r["model"] == model]
        print(
            f"{model}: format ok {sum(r['format_ok'] for r in mine)}/{len(mine)}, "
            f"prompt tokens match "
            f"{sum(r['prompt_tokens'] == r['local_tokens'] for r in mine)}/{len(mine)}, "
            f"cut off at max tokens {sum(r['finish_reason'] == 'length' for r in mine)}"
        )


def ask(question):
    models, waited = call("GET", "/v1/models")
    print(f"models: {[m['id'] for m in models.json()['data']]} (waited {waited:.0f}s)")
    result = complete(question, config.ADAPTER_NAME)
    choice = result["choices"][0]
    print(choice["text"])
    print(f"finish_reason: {choice['finish_reason']} | usage: {result['usage']}")


def main(command, argument):
    truststore.inject_into_ssl()
    if command == "ask":
        ask(argument)
    elif command == "check":
        print_summary(check(argument, Tokenizer.from_pretrained(config.MODEL_NAME)))
    else:
        raise ValueError(f"unknown command {command!r}: use ask or check")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
