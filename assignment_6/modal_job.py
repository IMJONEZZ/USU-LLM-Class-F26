"""Assignment 6 data preflight and (after approval) Modal LoRA jobs.

The preflight is CPU-only and never starts training. Training functions are
defined separately so the reviewed plan can be posted before any GPU run.
"""

from __future__ import annotations

import hashlib
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
    .add_local_file(
        "assignment_6/modal_job.py", "/root/project/modal_job_source.py", copy=True
    )
    .add_local_file(
        "assignment_6/own_examples.jsonl", "/root/project/own_examples.jsonl", copy=True
    )
    .add_local_file(
        "assignment_6/targeted_examples.jsonl",
        "/root/project/targeted_examples.jsonl",
        copy=True,
    )
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
def train(
    mode: str,
    max_length: int,
    run_name: str = "",
    data_source: str = "capybara",
    epochs: int = 0,
    learning_rate: float = 1e-4,
    batch_size: int = 2,
    grad_accum: int = 4,
    lora_rank: int = 16,
    lora_alpha: int = 0,
    warmup_ratio: float = 0.0,
    capybara_rows: int = 0,
    seed: int = 42,
    lora_dropout: float = 0.0,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    lr_scheduler_type: str = "linear",
    model_revision: str = "",
) -> dict[str, Any]:
    """Run a smoke, 500-example pilot, or full job after explicit approval."""

    import time

    import torch
    from datasets import Dataset, load_dataset
    from huggingface_hub import model_info
    from peft import LoraConfig, PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainerCallback,
        set_seed,
    )
    from trl import SFTConfig, SFTTrainer

    from src.a6_evaluation import (
        dataset_digest,
        development_metrics,
        split_manifest,
        stratified_family_split,
        validate_completion_batch,
    )
    from src.trainer import (
        APPROVED_FOLLOWUP_IDS,
        DATASET_ID,
        DATASET_REVISION,
        MODEL_ID,
        TrainingConfig,
        deduplicate_conversations,
        format_conversation,
        load_own_examples,
        prepare_candidate_pool,
        sample_with_followup_quota,
        save_training_result,
        training_arguments,
    )

    if mode not in {"smoke", "pilot", "final"}:
        raise ValueError("mode must be smoke, pilot, or final")
    aliases = {"mixed": "capybara+own", "own": "own"}
    parts = set(aliases.get(data_source, data_source).split("+"))
    if not parts or not parts <= {"capybara", "own", "targeted"}:
        raise ValueError("data_source parts must be capybara, own, or targeted")
    selected = []
    if "capybara" in parts:
        dataset = load_dataset(DATASET_ID, split="train", revision=DATASET_REVISION)
        selected = sample_with_followup_quota(
            prepare_candidate_pool(dataset),
            SELECTED_SIZE,
            FOLLOWUP_SIZE,
            approved_followup_ids=APPROVED_FOLLOWUP_IDS,
        )
        if capybara_rows:
            selected = selected[:capybara_rows]
    if "own" in parts:
        selected = selected + load_own_examples("/root/project/own_examples.jsonl")
    if "targeted" in parts:
        selected = selected + load_own_examples("/root/project/targeted_examples.jsonl")
    selected = deduplicate_conversations(selected)
    run = run_name or mode
    revision = model_info(MODEL_ID, revision=model_revision or "main").sha
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
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
    training_pool, development = stratified_family_split(selected, seed=seed)
    if mode == "smoke":
        training = training_pool[:16]
        development = development[:4]
    elif mode == "pilot":
        training = training_pool[:500]
    else:
        training = training_pool
    diagnostic_rows = [row for row in development if "evaluation" in row]
    config = TrainingConfig(
        max_length=max_length,
        output_dir=str(ARTIFACT_ROOT / run / "adapter"),
        epochs=1 if mode == "smoke" else (epochs or 3),
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        warmup_ratio=warmup_ratio,
        lora_dropout=lora_dropout,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        lr_scheduler_type=lr_scheduler_type,
        checkpoint_metric="eval_skill_match_accuracy"
        if diagnostic_rows
        else "eval_loss",
        seed=seed,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=revision,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha or config.lora_rank,
        lora_dropout=config.lora_dropout,
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

    def generate_rows(current_model: Any, rows: list[dict]) -> list[str]:
        was_training = current_model.training
        current_model.eval()
        answers = []
        try:
            for row in rows:
                inputs = tokenizer.apply_chat_template(
                    row["messages"][:-1],
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                ).to(current_model.get_input_embeddings().weight.device)
                with torch.inference_mode():
                    output = current_model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=192,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                answers.append(
                    tokenizer.decode(
                        output[0][inputs["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                    )
                )
        finally:
            current_model.train(was_training)
        return answers

    class DiagnosticSFTTrainer(SFTTrainer):
        """Select on generated development answers; record every diagnostic."""

        def evaluate(
            self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"
        ):
            metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
            if diagnostic_rows:
                generation_started = time.perf_counter()
                scores, details = development_metrics(
                    diagnostic_rows, generate_rows(self.model, diagnostic_rows)
                )
                scores = {
                    f"{metric_key_prefix}_{name}": value
                    for name, value in scores.items()
                }
                metrics.update(scores)
                self.log(scores)
                _write_json(
                    f"{run}/development_step_{self.state.global_step}.json",
                    {
                        "step": self.state.global_step,
                        "epoch": self.state.epoch,
                        "metrics": scores,
                        "answers": details,
                        "generation_seconds": time.perf_counter() - generation_started,
                        "scoring_note": "Conservative phrase/numeric diagnostics; inspect answers before final selection.",
                    },
                )
            return metrics

    trainer = DiagnosticSFTTrainer(
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
    # Check the actual TRL collator, not just the requested configuration.
    batch = trainer.data_collator(
        [trainer.train_dataset[i] for i in range(min(2, len(training)))]
    )
    batch_check = validate_completion_batch(batch)
    for index in range(min(2, len(training))):
        formatted = format_conversation(training[index]["messages"])
        prompt_length = len(
            tokenizer.apply_chat_template(
                formatted["prompt"], add_generation_prompt=True, tokenize=True
            )
        )
        if not torch.all(batch["labels"][index, :prompt_length] == -100):
            raise ValueError(
                "TRL collator did not mask the full inference-format prompt"
            )

    def adapter_fingerprint(current_model: Any) -> str:
        digest = hashlib.sha256()
        count = 0
        for name, parameter in current_model.named_parameters():
            if parameter.requires_grad:
                digest.update(name.encode())
                digest.update(parameter.detach().float().cpu().numpy().tobytes())
                count += parameter.numel()
        if not count:
            raise ValueError("no trainable adapter parameters")
        return digest.hexdigest()

    before_fingerprint = adapter_fingerprint(trainer.model)
    manifest = split_manifest(training, development)
    manifest.update(
        {
            "dataset_sha256": dataset_digest(selected),
            "model_revision": revision,
            "split_seed": seed,
            "split_policy": "15 percent by tag/skill; whole scenario families",
            "source_sha256": {
                str(path.relative_to("/root/project")): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sorted(
                    [
                        *Path("/root/project/src").glob("*.py"),
                        Path("/root/project/modal_job_source.py"),
                    ]
                )
            },
        }
    )
    _write_json(f"{run}/data_manifest.json", manifest)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output = trainer.train()
    torch.cuda.synchronize()
    wall_time = time.perf_counter() - started
    training_peak_memory = torch.cuda.max_memory_reserved() / 1024**3
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    after_fingerprint = adapter_fingerprint(trainer.model)
    if before_fingerprint == after_fingerprint:
        raise ValueError("training did not change the adapter weights")
    # A fresh base avoids accidentally attaching a second adapter to the trained model.
    check_rows = development[:2]
    expected_reload_answers = generate_rows(trainer.model, check_rows)
    reload_base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=revision,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    reloaded = PeftModel.from_pretrained(reload_base, config.output_dir)
    reload_answers = generate_rows(reloaded, check_rows)
    if reload_answers != expected_reload_answers:
        raise ValueError("saved adapter reload changed greedy development outputs")
    result = {
        "mode": mode,
        "run_name": run,
        "data_source": data_source,
        "training_rows": len(training),
        "development_rows": len(development),
        "wall_time_seconds": wall_time,
        "peak_gpu_memory_gib": training_peak_memory,
        "metrics": output.metrics,
        "log_history": trainer.state.log_history,
        "config": config.__dict__,
        "model_revision": revision,
        "dataset_sha256": manifest["dataset_sha256"],
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "pipeline_checks": {
            "collator": batch_check,
            "adapter_changed": True,
            "reload_outputs_match": True,
        },
        "estimated_full_seconds_from_pilot": (
            wall_time * math.ceil(len(training_pool) / 500) if mode == "pilot" else None
        ),
    }
    save_training_result(ARTIFACT_ROOT / run / "training_result.json", result)
    artifacts.commit()
    return json.loads(json.dumps(result))


@app.function(
    image=gpu_image,
    gpu="T4",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={MODEL_CACHE_PATH: model_cache, str(ARTIFACT_ROOT): artifacts},
    timeout=30 * 60,
    single_use_containers=True,
)
def answer_questions(
    adapter_dir: str = "final/adapter",
    output_name: str = "answers.json",
    skip_original: bool = False,
) -> dict[str, Any]:
    """Ask the original and fine-tuned model the ten questions, greedy decoding."""

    import torch
    from huggingface_hub import model_info
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.trainer import ASSIGNMENT_QUESTIONS, MODEL_ID

    artifacts.reload()
    adapter_path = ARTIFACT_ROOT / adapter_dir
    result_path = next(
        (
            parent / "training_result.json"
            for parent in (adapter_path.parent, adapter_path.parent.parent)
            if (parent / "training_result.json").exists()
        ),
        adapter_path.parent / "training_result.json",
    )
    training_result = (
        json.loads(result_path.read_text()) if result_path.exists() else {}
    )
    manifest_path = next(
        (
            parent / "data_manifest.json"
            for parent in (adapter_path.parent, adapter_path.parent.parent)
            if (parent / "data_manifest.json").exists()
        ),
        adapter_path.parent / "data_manifest.json",
    )
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    revision = (
        training_result.get("model_revision")
        or manifest.get("model_revision")
        or model_info(MODEL_ID).sha
    )
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=revision, torch_dtype=torch.float16, device_map="auto"
    )
    settings = {"do_sample": False, "max_new_tokens": 512, "system_prompt": None}

    def generate(current_model: Any) -> list[str]:
        current_model.eval()
        answers = []
        for question in ASSIGNMENT_QUESTIONS:
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(current_model.device)
            with torch.no_grad():
                output = current_model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=512,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = output[0][inputs["input_ids"].shape[1] :]
            answers.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
        return answers

    original = [] if skip_original else generate(model)
    tuned_model = PeftModel.from_pretrained(model, str(ARTIFACT_ROOT / adapter_dir))
    tuned = generate(tuned_model)
    result = {
        "model": MODEL_ID,
        "adapter": adapter_dir,
        "settings": settings,
        "model_revision": revision,
        "dataset_sha256": training_result.get("dataset_sha256")
        or manifest.get("dataset_sha256"),
        "legacy_revision_unrecorded": not bool(
            training_result.get("model_revision") or manifest.get("model_revision")
        ),
        "questions": list(ASSIGNMENT_QUESTIONS),
        "original_answers": original,
        "fine_tuned_answers": tuned,
    }
    _write_json(output_name, result)
    return result


@app.local_entrypoint()
def main(
    mode: str = "preflight",
    max_length: int = 0,
    run_name: str = "",
    data_source: str = "capybara",
    epochs: int = 0,
    learning_rate: float = 1e-4,
    batch_size: int = 2,
    grad_accum: int = 4,
    lora_rank: int = 16,
    lora_alpha: int = 0,
    warmup_ratio: float = 0.0,
    capybara_rows: int = 0,
    seed: int = 42,
    lora_dropout: float = 0.0,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    lr_scheduler_type: str = "linear",
    model_revision: str = "",
    skip_original: bool = False,
    adapter_dir: str = "final/adapter",
    output_name: str = "answers.json",
) -> None:
    if mode == "preflight":
        result = preflight.remote()
    elif mode == "answers":
        result = answer_questions.remote(adapter_dir, output_name, skip_original)
    else:
        if max_length < 1:
            raise ValueError("max_length must come from a reviewed preflight")
        result = train.remote(
            mode,
            max_length,
            run_name,
            data_source,
            epochs,
            learning_rate,
            batch_size,
            grad_accum,
            lora_rank,
            lora_alpha,
            warmup_ratio,
            capybara_rows,
            seed,
            lora_dropout,
            weight_decay,
            max_grad_norm,
            lr_scheduler_type,
            model_revision,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
