"""Send the ten Assignment 6 questions to the served base model and adapter.

Writes assignment_7/results/ten_question_check.json. The right-or-wrong verdicts
are judged afterwards against my Assignment 6 correctness guide.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

from src.inference import (
    API_KEY_ENV,
    BASE_URL_ENV,
    DEFAULT_BASE_URL,
    ask,
    list_models,
)

HERE = Path(__file__).parent
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER = "a6"


def protection_check(base_url: str) -> dict[str, int]:
    """Return the HTTP status for a request with no key and with a wrong key."""
    url = f"{base_url.rstrip('/')}/v1/models"
    return {
        "no_key": requests.get(url, timeout=60).status_code,
        "wrong_key": requests.get(
            url, headers={"Authorization": "Bearer wrong"}, timeout=60
        ).status_code,
    }


def speed_summary(rows: list[dict]) -> dict:
    """Latency medians and a rough tokens-per-second estimate for each model.

    The tokens-per-second figure is total completion tokens divided by total
    request time, so it includes the fixed cost of each request. It is only an
    estimate, and it is not fair across models whose answers differ in length.
    """
    summary: dict = {}
    latencies = {}
    for label in ("adapter", "base"):
        times = [r[label]["latency_seconds"] for r in rows]
        tokens = sum(r[label]["completion_tokens"] for r in rows)
        latencies[label] = times
        summary[f"median_latency_{label}_seconds"] = round(statistics.median(times), 3)
        summary[f"completion_tokens_{label}"] = tokens
        summary[f"estimated_tokens_per_second_{label}"] = round(tokens / sum(times), 1)
    summary["max_latency_seconds"] = max(latencies["adapter"] + latencies["base"])
    return summary


def main() -> None:
    api_key = os.environ[API_KEY_ENV]
    base_url = os.environ.get(BASE_URL_ENV, DEFAULT_BASE_URL)
    reference = json.loads((HERE / "a6_reference.json").read_text())
    started = datetime.now(UTC).isoformat(timespec="seconds")

    # The very first request may have to wake a scaled-to-zero server, so its
    # latency is recorded separately as the cold-start measurement.
    first = time.perf_counter()
    models = list_models(api_key=api_key, base_url=base_url)
    cold_start_seconds = round(time.perf_counter() - first, 1)

    rows = []
    for item in reference["questions"]:
        row = {"number": item["number"], "question": item["question"]}
        for label, model in (("adapter", ADAPTER), ("base", BASE_MODEL)):
            result = ask(
                item["question"], api_key=api_key, base_url=base_url, model=model
            )
            row[label] = {
                "answer": result["text"],
                "completion_tokens": result["usage"].get("completion_tokens"),
                "latency_seconds": result["latency_seconds"],
            }
        row["a6_answer"] = item["a6_answer"]
        row["a6_student_correct"] = item["a6_student_correct"]
        row["adapter_matches_a6_exactly"] = (
            row["adapter"]["answer"] == item["a6_answer"]
        )
        rows.append(row)
        print(f"Q{item['number']}: adapter match={row['adapter_matches_a6_exactly']}")

    summary = {
        "started_utc": started,
        "server_url": base_url,
        "models_listed": models,
        "first_request_seconds_cold_start_if_server_was_idle": cold_start_seconds,
        "protection_status_codes": protection_check(base_url),
        "exact_matches_with_a6": sum(r["adapter_matches_a6_exactly"] for r in rows),
        **speed_summary(rows),
        "settings": reference["settings"],
    }
    out = HERE / "results" / "ten_question_check.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False)
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
