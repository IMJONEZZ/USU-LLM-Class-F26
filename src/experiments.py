import os
import subprocess
import sys
from pathlib import Path

import modal
import truststore

from src import chart, config, stages

RUNS = ["m8_run1", "m8_run2", "m8_run3", "m8_run4", "m8_run5"]


def deploy(fast_boot):
    env = os.environ.copy()
    env["FAST_BOOT"] = "1" if fast_boot else "0"
    subprocess.run(["modal", "deploy", "-m", "src.modal_server"], env=env, check=True)


def wipe_volume(name):
    volume = modal.Volume.from_name(name)
    for entry in volume.listdir("/"):
        volume.remove_file(entry.path, recursive=True)


def require_cache(name, populated):
    count = stages.volume_file_count(name)
    if (count > 0) != populated:
        needed = "populated" if populated else "empty"
        raise ValueError(f"{name} has {count} files but this run needs it {needed}")


def _run_five(question):
    deploy(fast_boot=True)
    require_cache(config.VLLM_CACHE_VOLUME, populated=True)
    stages.run_and_save(question, RUNS[4])


def run_five(question):
    """Only run 5, on the state runs 1-4 left behind, then redraw the five-run chart."""
    missing = [
        n for n in RUNS[:4] if not (Path(config.RUNS_DIR) / f"{n}.json").exists()
    ]
    if missing:
        raise ValueError(f"run 5 needs the saved runs first, missing: {missing}")
    require_cache(config.HF_CACHE_VOLUME, populated=True)
    _run_five(question)
    chart.main(RUNS)


def run_experiments(question):
    try:
        stages.wait_for_no_containers()
        wipe_volume(config.HF_CACHE_VOLUME)
        wipe_volume(config.VLLM_CACHE_VOLUME)
        require_cache(config.HF_CACHE_VOLUME, populated=False)
        require_cache(config.VLLM_CACHE_VOLUME, populated=False)
        print("run 1 of 5: FAST_BOOT on, HF cache empty, vLLM cache empty")
        deploy(fast_boot=True)
        stages.run_and_save(question, RUNS[0])
        print("run 2 of 5: FAST_BOOT on, HF cache warm, vLLM cache empty")
        stages.run_and_save(question, RUNS[1])
        print("run 3 of 5: FAST_BOOT off, HF cache warm, vLLM cache empty")
        deploy(fast_boot=False)
        require_cache(config.VLLM_CACHE_VOLUME, populated=False)
        stages.run_and_save(question, RUNS[2])
        print("run 4 of 5: FAST_BOOT off, HF cache warm, vLLM cache populated")
        require_cache(config.VLLM_CACHE_VOLUME, populated=True)
        stages.run_and_save(question, RUNS[3])
        print("run 5 of 5: FAST_BOOT on, HF cache warm, vLLM cache populated")
        _run_five(question)
    finally:
        deploy(fast_boot=True)  # always leave the normal configuration deployed
    chart.main(RUNS)


def main(command, question):
    if command not in ("all", "run5"):
        raise ValueError(f"unknown command {command!r}: use all or run5")
    truststore.inject_into_ssl()
    if command == "all":
        run_experiments(question)
    else:
        run_five(question)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
