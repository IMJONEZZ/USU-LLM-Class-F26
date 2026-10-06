import subprocess
import threading
from time import perf_counter

import torch


class Timer:
    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed_seconds = perf_counter() - self.start


def read_gpu_usage():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    used_str, total_str, utilization_str = result.stdout.strip().split(",")
    used_mib = float(used_str)
    return {
        "memory_used_mib": used_mib,
        "memory_percent": used_mib / float(total_str) * 100,
        "utilization_percent": float(utilization_str),
    }


def summarize_gpu_samples(samples):
    utilizations = [sample["utilization_percent"] for sample in samples]
    return {
        "n_samples": len(samples),
        "peak_memory_mib": max(sample["memory_used_mib"] for sample in samples),
        "peak_memory_percent": max(sample["memory_percent"] for sample in samples),
        "mean_utilization_percent": sum(utilizations) / len(utilizations),
        "peak_utilization_percent": max(utilizations),
    }


class GpuMonitor:
    def __init__(self, interval=5.0):
        self.interval = interval
        self.samples = []
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._poll)

    def _poll(self):
        while not self._stop_event.is_set():
            self.samples.append(read_gpu_usage())
            self._stop_event.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop_event.set()
        self._thread.join()
        self.summary = summarize_gpu_samples(self.samples)


def peak_memory_allocated_mib():
    return torch.cuda.max_memory_allocated() / 1024**2
