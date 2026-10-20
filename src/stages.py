import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime

import modal
import truststore
from modal.volume import FileEntryType

from src import config, inference, modal_server

STAGES = ["container_boot", "imports", "vllm_startup", "routing_gap", "first_request"]

# vLLM's own duration lines (0.10.2 wording); a step that is not printed stays None
INNER_PATTERNS = {
    "download": r"Time spent downloading weights.*: ([\d.]+) seconds",
    "model_load": r"Model loading took [\d.]+ GiB and ([\d.]+) seconds",
    "torch_compile": r"torch\.compile takes ([\d.]+) s in total",
    "graph_capture": r"Graph capturing finished in ([\d.]+) secs?",
    "init_engine": r"init engine \(.*\) took ([\d.]+) seconds",
}
NESTED_LINES = ["Loading weights took"]  # already inside "Model loading took"
STAMP = re.compile(r"(?:INFO|WARNING|ERROR) (\d\d-\d\d \d\d:\d\d:\d\d)")


def stage_times(marks, sent_epoch, waited_s, ttft_s):
    answered = sent_epoch + waited_s  # when the proxy finally answered the request
    return {
        "container_boot": marks["import_epoch"] - sent_epoch,
        "imports": marks["launch_epoch"] - marks["import_epoch"],
        "vllm_startup": marks["ready_epoch"] - marks["launch_epoch"],
        "routing_gap": answered - marks["ready_epoch"],
        "first_request": ttft_s - waited_s,
    }


def wait_for_no_containers():
    while True:
        listing = subprocess.run(
            ["modal", "container", "list", "--json"],
            capture_output=True,
            text=True,
            check=False,
        )
        running = [
            c
            for c in json.loads(listing.stdout)
            if c["app_name"] == modal_server.app.name
        ]
        if not running:
            return
        time.sleep(config.RETRY_INTERVAL_SECONDS)


def read_logs():
    logs = subprocess.run(
        ["modal", "app", "logs", modal_server.app.name, "--tail", "300"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return logs.stdout


def find_marks(log_text):
    lines = [line for line in log_text.splitlines() if config.MARKS_TAG in line]
    if not lines:
        raise ValueError(f"no {config.MARKS_TAG} line in the app logs")
    return json.loads(lines[-1].partition(config.MARKS_TAG)[2])


def volume_file_count(name):
    entries = modal.Volume.from_name(name).listdir("/", recursive=True)
    return sum(1 for entry in entries if entry.type == FileEntryType.FILE)


def window_lines(log_text, marks):
    """The vLLM log lines stamped inside this start's launch-to-ready window (UTC)."""
    year = datetime.fromtimestamp(marks["launch_epoch"], UTC).year
    lines = []
    for line in log_text.splitlines():
        stamp = STAMP.search(line)
        if stamp:
            text = f"{year}-{stamp.group(1)} +0000"
            epoch = datetime.strptime(text, "%Y-%m-%d %H:%M:%S %z").timestamp()
            if marks["launch_epoch"] - 2 <= epoch <= marks["ready_epoch"] + 2:
                lines.append(line)
    return lines


def inner_steps(log_text, marks):
    lines = window_lines(log_text, marks)
    steps = {}
    for name, pattern in INNER_PATTERNS.items():
        found = [match for match in map(re.compile(pattern).search, lines) if match]
        steps[name] = float(found[-1].group(1)) if found else None
    return steps


def unrecognized_timing_lines(log_text, marks):
    known = list(INNER_PATTERNS.values()) + [re.escape(t) for t in NESTED_LINES]
    return [
        line
        for line in window_lines(log_text, marks)
        if re.search(r" (took|takes) ", line)
        and not any(re.search(pattern, line) for pattern in known)
    ]


def cold_start(question):
    wait_for_no_containers()
    cache = {
        "hf_cache_files": volume_file_count(config.HF_CACHE_VOLUME),
        "vllm_cache_files": volume_file_count(config.VLLM_CACHE_VOLUME),
    }
    sent_epoch = time.time()
    records = inference.stream_runs(question)
    cold = records[0]
    if cold["retries"] == 0:
        raise ValueError("the server was already warm, so this is not a cold start")
    log_text = read_logs()
    marks = find_marks(log_text)
    if marks["ready_epoch"] < sent_epoch:
        raise ValueError("the latest marks line is from an earlier start")
    cold.update(stage_times(marks, sent_epoch, cold["waited_s"], cold["ttft_s"]))
    cold.update(cache)
    cold["fast_boot"] = marks["fast_boot"]
    for name, seconds in inner_steps(log_text, marks).items():
        cold[f"vllm_{name}_s"] = seconds
    cold["unrecognized_timing_lines"] = unrecognized_timing_lines(log_text, marks)
    return records


def print_stages(record):
    for stage in STAGES:
        share = record[stage] / record["ttft_s"]
        print(f"{stage:15} {record[stage]:8.2f}s {share:5.0%}")
    print(f"{'total':15} {record['ttft_s']:8.2f}s")


def run_and_save(question, run):
    records = cold_start(question)
    inference.write_run(f"{run}.json", records)
    inference.print_stream(records)
    print_stages(records[0])


def main(question, run):
    truststore.inject_into_ssl()
    run_and_save(question, run)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
