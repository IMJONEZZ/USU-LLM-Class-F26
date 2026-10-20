import json
import subprocess
import sys
import time

import truststore

from src import config, inference, modal_server

STAGES = ["container_boot", "imports", "vllm_startup", "routing_gap", "first_request"]


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


def cold_start(question):
    wait_for_no_containers()
    sent_epoch = time.time()
    records = inference.stream_runs(question)
    cold = records[0]
    if cold["retries"] == 0:
        raise ValueError("the server was already warm, so this is not a cold start")
    marks = find_marks(read_logs())
    if marks["ready_epoch"] < sent_epoch:
        raise ValueError("the latest marks line is from an earlier start")
    cold.update(stage_times(marks, sent_epoch, cold["waited_s"], cold["ttft_s"]))
    return records


def print_stages(record):
    for stage in STAGES:
        share = record[stage] / record["ttft_s"]
        print(f"{stage:15} {record[stage]:8.2f}s {share:5.0%}")
    print(f"{'total':15} {record['ttft_s']:8.2f}s")


def main(question):
    truststore.inject_into_ssl()
    records = cold_start(question)
    inference.write_run("m7_coldstart.json", records)
    inference.print_stream(records)
    print_stages(records[0])


if __name__ == "__main__":
    main(sys.argv[1])
