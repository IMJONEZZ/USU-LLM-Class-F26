import json
from pathlib import Path

import modal

app = modal.App("resume-training")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
checkpoints = modal.Volume.from_name("resume-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.12.1",
        "torchao==0.18.0",
        "unsloth==2026.9.14",
        "transformers==5.5.0",
        "sentence-transformers==6.1.0",
        "datasets==4.3.0",
        "pandas==3.0.6",
        "numpy==2.4.6",
        "tqdm",
    )
    .env({"HF_HOME": "/hf_cache"})
    .add_local_dir("src", remote_path="/root/src")
)


@app.function(
    gpu="L40S",
    image=image,
    volumes={"/hf_cache": hf_cache, "/checkpoints": checkpoints},
    timeout=7200,
)
def run_training(
    run_name,
    overfit,
    overfit_train_examples,
    overfit_eval_examples,
    max_steps,
    batch_size,
    gradient_accumulation_steps,
):
    import unsloth  # isort: skip  # noqa: F401  (unsloth must be first)

    from importlib import metadata

    from datasets import load_dataset

    from src.config import MAX_STEPS, SEED
    from src.preprocessing import make_splits
    from src.trainer import build_text_dataset, build_trainer, load_model_for_training
    from src.utils import GpuMonitor, Timer, peak_memory_allocated_mib

    # and then we record the exact package versions this run used
    versions = {
        package: metadata.version(package)
        for package in ["torch", "unsloth", "transformers", "trl", "datasets"]
    }
    print("versions:", versions)

    # and then we rebuild the same cleaned, seeded splits as the baseline run
    resumes = load_dataset("Divyaamith/Kaggle-Resume")["train"].to_pandas()
    train_resumes, validation_resumes, _test_resumes = make_splits(resumes)
    categories = sorted(resumes["Category"].unique())
    if overfit:
        train_resumes = train_resumes.sample(
            n=overfit_train_examples, random_state=SEED
        )
        validation_resumes = validation_resumes.sample(
            n=overfit_eval_examples, random_state=SEED
        )
    print(
        "training on:", len(train_resumes), "| validating on:", len(validation_resumes)
    )

    # and then we load the 4-bit model with LoRA adapters and build the trainer
    model, tokenizer = load_model_for_training()
    train_dataset = build_text_dataset(train_resumes, categories, tokenizer.eos_token)
    eval_dataset = build_text_dataset(
        validation_resumes, categories, tokenizer.eos_token
    )
    overrides = {
        "max_steps": max_steps or (100 if overfit else MAX_STEPS),
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    overrides = {name: value for name, value in overrides.items() if value}
    trainer = build_trainer(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        f"/checkpoints/{run_name}",
        **overrides,
    )

    # and then we read one decoded training string by eye before spending GPU time
    print("first training string, last 200 characters:")
    print(repr(train_dataset[0]["text"][-200:]))

    # and then we train with the timer and GPU monitor on
    with Timer() as timer, GpuMonitor() as gpu_monitor:
        trainer.train()

    # and then we save the best adapter for the after-training evaluation
    final_dir = f"/checkpoints/{run_name}/final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    checkpoints.commit()
    hf_cache.commit()

    requested_steps = trainer.args.max_steps
    print(f"steps run: {trainer.state.global_step} of {requested_steps} requested")
    print("best checkpoint:", trainer.state.best_model_checkpoint)
    print(f"train seconds: {timer.elapsed_seconds:.1f}")
    print("gpu summary:", gpu_monitor.summary)
    print(f"peak torch memory allocated (MiB): {peak_memory_allocated_mib():.0f}")
    return {
        "run_name": run_name,
        "overfit": overfit,
        "train_examples": len(train_resumes),
        "validation_examples": len(validation_resumes),
        "requested_steps": requested_steps,
        "steps_run": trainer.state.global_step,
        "stopped_early": trainer.state.global_step < requested_steps,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "versions": versions,
        "train_seconds": timer.elapsed_seconds,
        "gpu_summary": gpu_monitor.summary,
        "peak_torch_memory_allocated_mib": peak_memory_allocated_mib(),
        "log_history": trainer.state.log_history,
    }


@app.local_entrypoint()
def main(
    overfit: bool = False,
    run_name: str = "",
    overfit_train_examples: int = 16,
    overfit_eval_examples: int = 32,
    max_steps: int = 0,
    batch_size: int = 0,
    gradient_accumulation_steps: int = 0,
):
    run_name = run_name or ("overfit" if overfit else "full")
    training_results = run_training.remote(
        run_name,
        overfit,
        overfit_train_examples,
        overfit_eval_examples,
        max_steps,
        batch_size,
        gradient_accumulation_steps,
    )
    Path("results").mkdir(exist_ok=True)
    Path(f"results/train_{run_name}.json").write_text(
        json.dumps(training_results, indent=2)
    )
