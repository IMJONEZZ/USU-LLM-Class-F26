"""Run Assignment 5 smoke, evaluation, and training jobs on Modal.

Examples:
    modal run assignment_5/modal_job.py --mode smoke
    modal run assignment_5/modal_job.py --mode baseline
    modal run assignment_5/modal_job.py --mode pilot
    modal run assignment_5/modal_job.py --mode final
    modal run assignment_5/modal_job.py --mode after
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

APP_NAME = "assignment-5-sst2"
ARTIFACT_ROOT = Path("/artifacts")
MODEL_CACHE_PATH = "/model_cache"

app = modal.App(APP_NAME)

# This current Unsloth stack supports TRL prompt/completion datasets.  The older
# stack in Modal's example requires a formatting function and cannot preserve
# completion-only loss for this dataset.  Python 3.14 CI remains GPU-free.
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "accelerate==1.9.0",
        "datasets==3.6.0",
        "hf-transfer==0.1.9",
        "huggingface_hub==0.34.2",
        "peft==0.18.0",
        "transformers==4.56.2",
        "trl==0.24.0",
        "unsloth[cu128-torch270]==2026.9.14",
        "unsloth_zoo==2026.9.9",
    )
    .env(
        {
            "HF_HOME": MODEL_CACHE_PATH,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONPATH": "/root/project",
        }
    )
    .add_local_dir("src", "/root/project/src", copy=True)
    .workdir("/root/project")
)

model_cache = modal.Volume.from_name("a5-model-cache", create_if_missing=True)
artifacts = modal.Volume.from_name("a5-artifacts", create_if_missing=True)


def _runtime_metadata(torch_module: Any) -> dict[str, Any]:
    """Collect versions and hardware details needed in the results comment."""

    import datasets
    import transformers
    import trl
    import unsloth

    device = torch_module.cuda.get_device_properties(0)
    return {
        "torch_version": torch_module.__version__,
        "cuda_version": torch_module.version.cuda,
        "datasets_version": datasets.__version__,
        "transformers_version": transformers.__version__,
        "trl_version": trl.__version__,
        "unsloth_version": unsloth.__version__,
        "gpu_name": device.name,
        "gpu_total_memory_gib": device.total_memory / 1024**3,
    }


def _show_gpu() -> None:
    """Print a visible point-in-time GPU snapshot for run evidence."""

    import subprocess

    subprocess.run(["nvidia-smi"], check=True)


def _write_json(name: str, value: dict[str, Any]) -> None:
    destination = ARTIFACT_ROOT / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    artifacts.commit()


def _json_safe(value: dict[str, Any]) -> dict[str, Any]:
    """Remove library-specific scalar subclasses before Modal serializes output."""

    return json.loads(json.dumps(value))


@app.function(
    image=gpu_image,
    gpu="T4",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={MODEL_CACHE_PATH: model_cache, str(ARTIFACT_ROOT): artifacts},
    timeout=6 * 60 * 60,
    single_use_containers=True,
)
def smoke() -> dict[str, Any]:
    """Exercise model loading and one tiny end-to-end training run."""

    # Unsloth must be imported before Transformers and TRL apply their patches.
    import unsloth  # noqa: F401, I001
    import torch

    from src.trainer import TrainingConfig, train_sst2

    _show_gpu()
    config = TrainingConfig(
        development_fraction=0.0001,
        pilot_size=16,
        epochs=1,
        logging_steps=1,
        evaluation_steps=1,
        output_dir=str(ARTIFACT_ROOT / "smoke" / "adapter"),
        result_path=str(ARTIFACT_ROOT / "smoke" / "training_result.json"),
    )
    result = train_sst2(config, pilot=True)
    payload = {
        "status": "ok",
        "runtime": _runtime_metadata(torch),
        "training": result.to_dict(),
    }
    _show_gpu()
    _write_json("smoke/summary.json", payload)
    return _json_safe(payload)


@app.function(
    image=gpu_image,
    gpu="T4",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={MODEL_CACHE_PATH: model_cache, str(ARTIFACT_ROOT): artifacts},
    timeout=6 * 60 * 60,
    single_use_containers=True,
)
def evaluate(checkpoint: str = "base") -> dict[str, Any]:
    """Evaluate the untouched base model or the final saved adapter on SST-2."""

    import unsloth  # noqa: F401, I001
    import torch
    from unsloth import FastLanguageModel

    from src.evaluator import Evaluator, LlamaSST2Predictor, load_sst2_validation
    from src.trainer import MODEL_ID, TrainingConfig

    if checkpoint not in {"base", "final"}:
        raise ValueError("checkpoint must be 'base' or 'final'")
    model_path = (
        MODEL_ID if checkpoint == "base" else str(ARTIFACT_ROOT / "final" / "adapter")
    )
    if checkpoint == "final" and not Path(model_path).exists():
        raise FileNotFoundError("Run the final training job before after-evaluation")

    _show_gpu()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=TrainingConfig.max_sequence_length,
        dtype=None,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)
    texts, labels = load_sst2_validation()
    predictor = LlamaSST2Predictor(model, tokenizer, torch, batch_size=16)
    result = Evaluator(predictor, seed=TrainingConfig.seed).evaluate(texts, labels)
    payload = {
        "checkpoint": checkpoint,
        "model_path": model_path,
        "runtime": _runtime_metadata(torch),
        "evaluation": result.to_dict(),
    }
    _show_gpu()
    _write_json(f"evaluation_{checkpoint}.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return _json_safe(payload)


@app.function(
    image=gpu_image,
    gpu="T4",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={MODEL_CACHE_PATH: model_cache, str(ARTIFACT_ROOT): artifacts},
    timeout=6 * 60 * 60,
    single_use_containers=True,
)
def train(pilot: bool = True) -> dict[str, Any]:
    """Run the planned pilot or a fresh full training job."""

    import unsloth  # noqa: F401, I001
    import torch

    from src.trainer import TrainingConfig, train_sst2

    run_name = "pilot" if pilot else "final"
    _show_gpu()
    config = TrainingConfig(
        output_dir=str(ARTIFACT_ROOT / run_name / "adapter"),
        result_path=str(ARTIFACT_ROOT / run_name / "training_result.json"),
    )
    result = train_sst2(config, pilot=pilot)
    payload = {
        "run": run_name,
        "runtime": _runtime_metadata(torch),
        "training": result.to_dict(),
    }
    _show_gpu()
    _write_json(f"{run_name}/summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return _json_safe(payload)


@app.local_entrypoint()
def main(mode: str = "smoke") -> None:
    """Dispatch exactly one paid GPU job at a time."""

    if mode == "smoke":
        result = smoke.remote()
    elif mode == "baseline":
        result = evaluate.remote("base")
    elif mode == "pilot":
        result = train.remote(True)
    elif mode == "final":
        result = train.remote(False)
    elif mode == "after":
        result = evaluate.remote("final")
    else:
        raise ValueError("mode must be smoke, baseline, pilot, final, or after")
    print(json.dumps(result, indent=2, sort_keys=True))
