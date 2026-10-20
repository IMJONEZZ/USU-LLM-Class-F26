import json
import subprocess
import sys
import time

import modal
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


def read_marks():
    volume = modal.Volume.from_name(config.HF_CACHE_VOLUME)
    return json.loads(b"".join(volume.read_file(config.MARKS_FILE)))


def cold_start(question):
    wait_for_no_containers()
    sent_epoch = time.time()
    run = inference.stream(question, config.ADAPTER_NAME)
    if run["retries"] == 0:
        raise ValueError("the server was already warm, so this is not a cold start")
    record = stage_times(read_marks(), sent_epoch, run["waited_s"], run["ttft_s"])
    record["total_s"] = run["ttft_s"]
    return record


def print_stages(record):
    for stage in STAGES:
        share = record[stage] / record["total_s"]
        print(f"{stage:15} {record[stage]:8.2f}s {share:5.0%}")
    print(f"{'total':15} {record['total_s']:8.2f}s")


def main(question):
    truststore.inject_into_ssl()
    record = cold_start(question)
    inference.write_run("m7_stages.json", [record])
    print_stages(record)


if __name__ == "__main__":
    main(sys.argv[1])
