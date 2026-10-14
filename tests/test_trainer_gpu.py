"""Opt-in real Unsloth integration, run as a subprocess for correct import order."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("RUN_A5_GPU") != "1" or not torch.cuda.is_available(),
    reason="Requires explicit RUN_A5_GPU=1, prepared real data and working CUDA",
)
def test_unsloth_real_optimizer_steps(tmp_path):
    prepared = Path(os.environ.get("A5_PREPARED", "data/assignment5/prepared"))
    output = tmp_path / "feasibility"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.trainer",
            "feasibility",
            "--prepared",
            str(prepared),
            "--output",
            str(output),
        ],
        check=True,
    )
    result = json.loads((output / "training.json").read_text())
    assert result["history"][0]["optimizer_steps"] == 2
    assert result["peak_allocated_bytes"] > 0
    assert (output / "best/adapter_model.safetensors").is_file()
    assert (output / "success.json").is_file()
