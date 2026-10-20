import json
import os
import sys
import time
from itertools import pairwise
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
    write_run("m4_correctness.json", records)
    return records


def write_run(name, records):
    rounded = [
        {k: round(v, 2) if isinstance(v, float) else v for k, v in record.items()}
        for record in records
    ]
    Path(config.RUNS_DIR).mkdir(exist_ok=True)
    (Path(config.RUNS_DIR) / name).write_text(json.dumps(rounded, indent=2))


def stream_stats(arrivals):
    """arrivals: seconds from sending the request to each streamed token chunk."""
    gaps = [later - earlier for earlier, later in pairwise(arrivals)]
    return {
        "ttft_s": arrivals[0],
        "chunks": len(arrivals),
        "rate_tok_s": (len(arrivals) - 1) / (arrivals[-1] - arrivals[0]),
        "max_gap_s": max(gaps),
    }


def stream(question, model):
    body = completion_request(question, model)
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    sent = time.perf_counter()
    response, waited, retries = call("POST", "/v1/completions", json=body, stream=True)
    arrivals, usage, text = [], None, ""
    for line in response.iter_lines(decode_unicode=True):
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[len("data: ") :])
            if chunk["choices"]:
                arrivals.append(time.perf_counter() - sent)
                text += chunk["choices"][0]["text"]
            else:
                usage = chunk["usage"]
    result = stream_stats(arrivals)
    result["completion_tokens"] = usage["completion_tokens"]
    result["waited_s"] = waited
    result["retries"] = retries
    result["num_words"] = len(text.split())
    result["wpm"] = result["num_words"] / (arrivals[-1] - arrivals[0]) * 60
    return result


def stream_runs(question):
    """One stream; if it had to wait out a cold start (a 503 retry), a second one measures the warm server."""
    first = stream(question, config.ADAPTER_NAME)
    cold = first["retries"] > 0
    first["run"] = "cold start" if cold else "warm"
    records = [first]
    if cold:
        second = stream(question, config.ADAPTER_NAME)
        second["run"] = "warm"
        records.append(second)
    records[-1]["gate_pass"] = records[-1]["rate_tok_s"] >= config.MIN_TOKENS_PER_SEC
    return records


def print_stream(records):
    for r in records:
        print(
            f"{r['run']}: first token {r['ttft_s']:.2f}s | {r['rate_tok_s']:.1f} tok/s "
            f"after the first | {r['chunks']} chunks (server counted "
            f"{r['completion_tokens']}) | {r['wpm']:.0f} words/min | longest gap "
            f"{r['max_gap_s'] * 1000:.0f} ms | waited {r['waited_s']:.0f}s through "
            f"{r['retries']} 503s"
        )
    warm = records[-1]
    verdict = "PASS" if warm["gate_pass"] else "FAIL"
    print(
        f"gate: {verdict}, warm rate {warm['rate_tok_s']:.2f} tok/s "
        f"(minimum {config.MIN_TOKENS_PER_SEC})"
    )


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
    elif command == "check":
        print_summary(check(argument, Tokenizer.from_pretrained(config.MODEL_NAME)))
    elif command == "stream":
        records = stream_runs(argument)
        write_run("m5_stream.json", records)
        print_stream(records)
    else:
        raise ValueError(f"unknown command {command!r}: use ask, check or stream")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
