"""Assignment 6 data preflight and (after approval) Modal LoRA jobs.

The preflight is CPU-only and never starts training. Training functions are
defined separately so the reviewed plan can be posted before any GPU run.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import modal

APP_NAME = "assignment-6-finetune"
ARTIFACT_ROOT = Path("/artifacts")
MODEL_CACHE_PATH = "/model_cache"
SELECTED_SIZE = 1_000
FOLLOWUP_SIZE = 40

app = modal.App(APP_NAME)
data_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "datasets==3.6.0",
        "huggingface_hub==0.34.2",
        "jinja2==3.1.6",
        "transformers==4.56.2",
    )
    .env({"HF_HOME": MODEL_CACHE_PATH, "PYTHONPATH": "/root/project"})
    .add_local_dir("src", "/root/project/src", copy=True)
    .workdir("/root/project")
)
gpu_image = data_image.uv_pip_install(
    "accelerate==1.9.0",
    "peft==0.18.0",
    "torch==2.7.0",
    "trl==0.24.0",
)
model_cache = modal.Volume.from_name("a6-model-cache", create_if_missing=True)
artifacts = modal.Volume.from_name("a6-artifacts", create_if_missing=True)


def _write_json(name: str, value: dict[str, Any]) -> None:
    path = ARTIFACT_ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    artifacts.commit()


def _percentile(values: list[int], fraction: float) -> int:
    values = sorted(values)
    return values[int(fraction * (len(values) - 1))]


@app.function(
    image=data_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={MODEL_CACHE_PATH: model_cache, str(ARTIFACT_ROOT): artifacts},
    timeout=20 * 60,
)
def preflight() -> dict[str, Any]:
    """Check a reproducible candidate spread and exact model-token lengths."""

    from datasets import load_dataset
    from transformers import AutoTokenizer

    from src.trainer import (
        APPROVED_FOLLOWUP_IDS,
        DATASET_ID,
        DATASET_REVISION,
        MODEL_ID,
        prepare_candidate_pool,
        sample_with_followup_quota,
    )

    dataset = load_dataset(DATASET_ID, split="train", revision=DATASET_REVISION)
    if len(dataset) != 15_806:
        raise ValueError(f"unexpected dataset size: {len(dataset)}")
    candidates = prepare_candidate_pool(dataset)
    selected = sample_with_followup_quota(
        candidates,
        SELECTED_SIZE,
        FOLLOWUP_SIZE,
        approved_followup_ids=APPROVED_FOLLOWUP_IDS,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    lengths = [
        len(tokenizer.apply_chat_template(row["messages"], tokenize=True))
        for row in selected
    ]
    manifest = {
        "dataset": DATASET_ID,
        "revision": DATASET_REVISION,
        "model_tokenizer": MODEL_ID,
        "selection_seed": 42,
        "selected_size": SELECTED_SIZE,
        "followup_size": FOLLOWUP_SIZE,
        "rows": [
            {
                "id": row["id"],
                "source": row["source"],
                "format": row["format"],
                "words": row["word_count"],
                "tokens": length,
            }
            for row, length in zip(selected, lengths, strict=True)
        ],
    }
    _write_json("preflight_candidate_manifest.json", manifest)
    summary = {
        "total_rows": len(dataset),
        "filtered_unique_candidates": len(candidates),
        "sampled_rows": len(selected),
        "formats": dict(Counter(row["format"] for row in selected)),
        "sources": dict(Counter(row["source"] for row in selected)),
        "tokens": {
            "median": _percentile(lengths, 0.5),
            "p90": _percentile(lengths, 0.9),
            "p95": _percentile(lengths, 0.95),
            "max": max(lengths),
        },
        "over_1024": sum(length > 1024 for length in lengths),
        "over_1536": sum(length > 1536 for length in lengths),
        "over_2048": sum(length > 2048 for length in lengths),
        "longest_rows": sorted(
            (
                (length, row["id"])
                for row, length in zip(selected, lengths, strict=True)
            ),
            reverse=True,
        )[:8],
        "manifest": "preflight_candidate_manifest.json",
    }
    _write_json("preflight_summary.json", summary)
    return summary


@app.function(
    image=gpu_image,
    gpu="T4",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={MODEL_CACHE_PATH: model_cache, str(ARTIFACT_ROOT): artifacts},
    timeout=6 * 60 * 60,
    single_use_containers=True,
)
def train(mode: str, max_length: int) -> dict[str, Any]:
    """Run a smoke, 500-example pilot, or full job after explicit approval."""

    import time

    import torch
    from datasets import Dataset, load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    from src.trainer import (
        APPROVED_FOLLOWUP_IDS,
        DATASET_ID,
        DATASET_REVISION,
        MODEL_ID,
        TrainingConfig,
        format_conversation,
        grouped_train_development_split,
        prepare_candidate_pool,
        sample_with_followup_quota,
        save_training_result,
        training_arguments,
    )

    if mode not in {"smoke", "pilot", "final"}:
        raise ValueError("mode must be smoke, pilot, or final")
    dataset = load_dataset(DATASET_ID, split="train", revision=DATASET_REVISION)
    selected = sample_with_followup_quota(
        prepare_candidate_pool(dataset),
        SELECTED_SIZE,
        FOLLOWUP_SIZE,
        approved_followup_ids=APPROVED_FOLLOWUP_IDS,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    lengths = [
        len(tokenizer.apply_chat_template(row["messages"], tokenize=True))
        for row in selected
    ]
    too_long = [
        row["id"]
        for row, length in zip(selected, lengths, strict=True)
        if length > max_length
    ]
    if too_long:
        raise ValueError(
            f"token cap would truncate {len(too_long)} examples; first IDs: {too_long[:5]}"
        )
    training_pool, development = grouped_train_development_split(selected)
    if mode == "smoke":
        training = training_pool[:16]
        development = development[:4]
    elif mode == "pilot":
        training = training_pool[:500]
    else:
        training = training_pool
    config = TrainingConfig(
        max_length=max_length,
        output_dir=str(ARTIFACT_ROOT / mode / "adapter"),
        epochs=1 if mode == "smoke" else 2,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_rank,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    class CommitOnSave(TrainerCallback):
        """Persist each checkpoint to the volume so a crash loses little."""

        def on_save(self, args, state, control, **kwargs):
            artifacts.commit()

    trainer = SFTTrainer(
        callbacks=[CommitOnSave()],
        model=model,
        args=SFTConfig(**training_arguments(config, len(training))),
        train_dataset=Dataset.from_list(
            [format_conversation(row["messages"]) for row in training]
        ),
        eval_dataset=Dataset.from_list(
            [format_conversation(row["messages"]) for row in development]
        ),
        processing_class=tokenizer,
        peft_config=lora,
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output = trainer.train()
    torch.cuda.synchronize()
    wall_time = time.perf_counter() - started
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    result = {
        "mode": mode,
        "training_rows": len(training),
        "development_rows": len(development),
        "wall_time_seconds": wall_time,
        "peak_gpu_memory_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "metrics": output.metrics,
        "log_history": trainer.state.log_history,
        "config": config.__dict__,
        "estimated_full_seconds_from_pilot": (
            wall_time * math.ceil(len(training_pool) / 500) if mode == "pilot" else None
        ),
    }
    save_training_result(ARTIFACT_ROOT / mode / "training_result.json", result)
    artifacts.commit()
    return json.loads(json.dumps(result))


@app.local_entrypoint()
def main(mode: str = "preflight", max_length: int = 0) -> None:
    if mode == "preflight":
        result = preflight.remote()
    else:
        if max_length < 1:
            raise ValueError("max_length must come from a reviewed preflight")
        result = train.remote(mode, max_length)
    print(json.dumps(result, indent=2, sort_keys=True))
