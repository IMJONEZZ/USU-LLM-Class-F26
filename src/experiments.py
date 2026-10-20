import os
import subprocess
import sys

import modal
import truststore

from src import chart, config, stages

RUNS = ["m8_run1", "m8_run2", "m8_run3", "m8_run4"]


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


def run_experiments(question):
    try:
        stages.wait_for_no_containers()
        wipe_volume(config.HF_CACHE_VOLUME)
        wipe_volume(config.VLLM_CACHE_VOLUME)
        require_cache(config.HF_CACHE_VOLUME, populated=False)
        require_cache(config.VLLM_CACHE_VOLUME, populated=False)
        print("run 1 of 4: FAST_BOOT on, HF cache empty, vLLM cache empty")
        deploy(fast_boot=True)
        stages.run_and_save(question, RUNS[0])
        print("run 2 of 4: FAST_BOOT on, HF cache warm, vLLM cache empty")
        stages.run_and_save(question, RUNS[1])
        print("run 3 of 4: FAST_BOOT off, HF cache warm, vLLM cache empty")
        deploy(fast_boot=False)
        require_cache(config.VLLM_CACHE_VOLUME, populated=False)
        stages.run_and_save(question, RUNS[2])
        print("run 4 of 4: FAST_BOOT off, HF cache warm, vLLM cache populated")
        require_cache(config.VLLM_CACHE_VOLUME, populated=True)
        stages.run_and_save(question, RUNS[3])
    finally:
        deploy(fast_boot=True)  # always leave the normal configuration deployed
    chart.main(RUNS)


def main(question):
    truststore.inject_into_ssl()
    run_experiments(question)


if __name__ == "__main__":
    main(sys.argv[1])
