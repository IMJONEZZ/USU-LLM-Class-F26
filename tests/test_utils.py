import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.utils import (
    GpuMonitor,
    Timer,
    peak_memory_allocated_mib,
    read_gpu_usage,
    summarize_gpu_samples,
)

FAKE_READING = {
    "memory_used_mib": 1000.0,
    "memory_percent": 10.0,
    "utilization_percent": 20.0,
}


@patch("src.utils.perf_counter", side_effect=[10.0, 12.5])
def test_timer_measures_elapsed_seconds(mock_perf_counter):
    with Timer() as timer:
        pass

    assert timer.elapsed_seconds == pytest.approx(2.5)


@patch("src.utils.subprocess.run")
def test_read_gpu_usage_parses_nvidia_smi_output(mock_run):
    mock_run.return_value = MagicMock(stdout="5000, 20000, 37\n")

    reading = read_gpu_usage()

    assert reading == {
        "memory_used_mib": 5000.0,
        "memory_percent": 25.0,
        "utilization_percent": 37.0,
    }
    command = mock_run.call_args.args[0]
    assert command[0] == "nvidia-smi"
    assert "--query-gpu=memory.used,memory.total,utilization.gpu" in command


def test_summarize_gpu_samples_hand_checked():
    samples = [
        {
            "memory_used_mib": 1000.0,
            "memory_percent": 10.0,
            "utilization_percent": 20.0,
        },
        {
            "memory_used_mib": 2000.0,
            "memory_percent": 20.0,
            "utilization_percent": 60.0,
        },
    ]

    assert summarize_gpu_samples(samples) == {
        "n_samples": 2,
        "peak_memory_mib": 2000.0,
        "peak_memory_percent": 20.0,
        "mean_utilization_percent": 40.0,
        "peak_utilization_percent": 60.0,
    }


@patch("src.utils.read_gpu_usage", return_value=FAKE_READING)
def test_gpu_monitor_samples_while_running_and_stops_on_exit(mock_read):
    with GpuMonitor(interval=0.01) as monitor:
        time.sleep(0.1)
    calls_after_exit = mock_read.call_count
    time.sleep(0.05)

    assert mock_read.call_count == calls_after_exit
    assert monitor.summary["n_samples"] >= 1
    assert monitor.summary["peak_memory_mib"] == 1000.0


@patch("src.utils.torch.cuda.max_memory_allocated", return_value=3 * 1024**2)
def test_peak_memory_allocated_mib_converts_bytes(mock_max_memory):
    assert peak_memory_allocated_mib() == pytest.approx(3.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_read_gpu_usage_real_gpu():
    reading = read_gpu_usage()

    assert 0 <= reading["memory_percent"] <= 100
    assert 0 <= reading["utilization_percent"] <= 100
