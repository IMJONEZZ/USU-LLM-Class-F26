import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config

app = modal.App("trick-question-cot")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
checkpoints = modal.Volume.from_name("trick-checkpoints", create_if_missing=True)

# the same package set that ran on Modal in the earlier project (image layers are cached)
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

_PACKAGES = ["torch", "torchao", "unsloth", "transformers", "datasets", "numpy"]


@app.function(
    gpu=config.GPU_TYPE,
    image=image,
    volumes={"/hf_cache": hf_cache, "/checkpoints": checkpoints},
    timeout=3600,
)
def evaluate_on_gpu(
    rows, prompt_style, model_path, batch_size, max_seq_length, max_new_tokens
):
    import unsloth  # isort: skip  # noqa: F401  (unsloth must be imported first)
    from unsloth import FastLanguageModel

    from src import eval as evaluation

    # and then we record the exact package versions this run used
    versions = evaluation.package_versions(_PACKAGES)
    print("versions:", versions)

    # and then we load the model in 4-bit, the same way training will
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        device_map={"": 0},
    )
    FastLanguageModel.for_inference(model)
    model.generation_config.max_length = None

    make_prompt = (
        evaluation.baseline_prompt
        if prompt_style == "baseline"
        else evaluation.finetuned_prompt
    )
    generate_fn = evaluation.make_generate_fn(
        model, tokenizer, max_new_tokens, batch_size
    )

    # and then we read two raw generations by eye before trusting any metric
    for row, text in zip(
        rows[:2], generate_fn([make_prompt(row) for row in rows[:2]]), strict=False
    ):
        print(f"expected={row['answer']!r}\ngenerated={text!r}\n")

    # and then we run the full evaluation with the timer and GPU monitor on
    with evaluation.Timer() as timer, evaluation.GpuMonitor() as gpu_monitor:
        results, metrics = evaluation.evaluate(rows, generate_fn, make_prompt)
    print(f"eval seconds: {timer.seconds:.1f}")
    hf_cache.commit()
    return {
        "results": results,
        "metrics": metrics,
        "versions": versions,
        "eval_seconds": timer.seconds,
        "gpu_summary": gpu_monitor.summary(),
    }


@app.local_entrypoint()
def eval_split(
    split: str = "eval",
    prompt_style: str = "baseline",
    adapter_path: str = "",
    output: str = "",
    force: bool = False,
):
    from src import data
    from src import eval as evaluation

    if prompt_style not in ("baseline", "finetuned"):
        raise ValueError(
            f"prompt_style must be baseline or finetuned: {prompt_style!r}"
        )
    if prompt_style == "finetuned" and not adapter_path:
        raise ValueError("--adapter-path is needed for the finetuned model")
    output_path = evaluation.default_output_path(split, prompt_style, output)
    evaluation.ensure_split_allowed(split, output_path, force)
    rows = evaluation.load_split_rows(split)
    if split == "test":
        data.check_test_set_hash(config.TEST_SET_PATH, config.TEST_SET_HASH_PATH)

    # and then we check every prompt fits, for free, before any GPU time is spent
    make_prompt = (
        evaluation.baseline_prompt
        if prompt_style == "baseline"
        else evaluation.finetuned_prompt
    )
    prompts = [make_prompt(row) for row in rows]
    tokenizer = data.load_tokenizer(config.MODEL_NAME)
    room = config.EVAL_MAX_SEQ_LENGTH - config.EVAL_MAX_NEW_TOKENS
    evaluation.check_prompt_length(prompts, tokenizer, room)
    longest = max(
        len(tokenizer.encode(p, add_special_tokens=True).ids) for p in prompts
    )
    print(
        f"{len(rows)} rows from the {split} split, {prompt_style} prompt: longest "
        f"prompt {longest} tokens (room for {room}); GPU {config.GPU_TYPE}"
    )

    remote = evaluate_on_gpu.remote(
        rows,
        prompt_style,
        adapter_path or config.MODEL_NAME,
        config.EVAL_BATCH_SIZE,
        config.EVAL_MAX_SEQ_LENGTH,
        config.EVAL_MAX_NEW_TOKENS,
    )
    result = evaluation.build_result(
        split=split,
        metrics=remote["metrics"],
        rows=rows,
        batch_size=config.EVAL_BATCH_SIZE,
        max_seq_length=config.EVAL_MAX_SEQ_LENGTH,
        versions=remote["versions"],
        eval_seconds=remote["eval_seconds"],
        gpu_summary=remote["gpu_summary"],
        prompt_style=prompt_style,
        max_new_tokens=config.EVAL_MAX_NEW_TOKENS,
    )
    evaluation.save_result(output_path, result)
    outputs_path = config.DATA_DIR / f"{output_path.stem}_outputs.jsonl"
    data.write_jsonl(outputs_path, remote["results"])
    metrics = remote["metrics"]
    print(f"saved {output_path} and the per-row outputs in {outputs_path}")
    print(
        f"accuracy {metrics['accuracy']}, format compliance "
        f"{metrics['format_compliance']}, n={metrics['n']}"
    )
    for category, cell in metrics["per_category"].items():
        print(f"  {category}: {cell['correct']}/{cell['n']} ({cell['accuracy']})")


@app.function(
    gpu=config.GPU_TYPE,
    image=image,
    volumes={"/hf_cache": hf_cache, "/checkpoints": checkpoints},
    timeout=7200,
)
def train_on_gpu(train_rows, eval_rows, round_n, probe, overrides):
    import unsloth  # isort: skip  # noqa: F401  (unsloth must be imported first)

    from src import eval as evaluation
    from src import train as training

    # and then we record the exact package versions this run used
    versions = evaluation.package_versions(
        ["torch", "unsloth", "transformers", "trl", "datasets", "peft"]
    )
    print("versions:", versions)

    # and then we load the 4-bit model with LoRA adapters, the same way the baseline did
    model, tokenizer = training.load_model_for_training()
    if tokenizer.eos_token != config.EOS_TOKEN:
        raise ValueError(
            f"tokenizer end token {tokenizer.eos_token!r} is not {config.EOS_TOKEN!r}"
        )
    train_dataset = training.build_text_dataset(train_rows, tokenizer.eos_token)
    eval_dataset = training.build_text_dataset(eval_rows, tokenizer.eos_token)

    # and then we hook the per-epoch evaluation (generation on the eval set) into training
    generate_fn = evaluation.make_generate_fn(model, tokenizer)
    evaluator = training.EpochEvaluator(eval_rows, generate_fn)
    adapter_dir = config.adapter_path(round_n, root="/checkpoints")
    trainer = training.build_trainer(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        str(adapter_dir),
        extra_callbacks=[training.make_epoch_eval_callback(evaluator)],
        **training.clean_overrides(overrides),
    )

    # and then we check, on the real trainer, that only the chain and the end token are trained
    checked = min(4, len(train_rows))
    examples = [trainer.train_dataset[i] for i in range(checked)]
    training.check_masking(tokenizer, train_rows[:checked], examples)
    print(
        f"masking checked on {checked} real examples: only the chain and end token are trained"
    )
    print("first example, what is trained on:")
    print(repr(training.expected_trained_text(tokenizer, train_rows[0])[:200]))

    # and then we train with the timer and GPU monitor on
    with evaluation.Timer() as timer, evaluation.GpuMonitor() as gpu_monitor:
        trainer.train()
    state = trainer.state
    print(f"epochs run: {state.epoch} | steps run: {state.global_step}")
    print("best checkpoint:", state.best_model_checkpoint)

    # and then we keep the best adapter for the next round and the final evaluation
    if not probe:
        training.save_adapter(trainer, tokenizer, str(adapter_dir))
        checkpoints.commit()
    hf_cache.commit()
    peak = evaluation._torch_peak_mib()
    record = training.build_training_record(
        round_n=round_n,
        trainer_state=state,
        train_rows=train_rows,
        eval_rows=eval_rows,
        requested_epochs=trainer.args.num_train_epochs,
        epoch_results=evaluator.epochs,
        versions=versions,
        train_seconds=timer.seconds,
        gpu_summary=gpu_monitor.summary(),
        peak_torch_memory_allocated_mib=peak,
        hyperparams=training.hyperparameters(**training.clean_overrides(overrides)),
    )
    print(f"train seconds: {timer.seconds:.1f}; peak torch memory (MiB): {peak}")
    return {"record": record, "masking_checked": checked}


def _sentinel(value):
    return None if value < 0 else value


@app.local_entrypoint()
def train_round(
    probe: bool = False,
    round_n: int = 0,
    epochs: int = -1,
    batch_size: int = -1,
    grad_accum: int = -1,
    seed: int = -1,
):
    import json

    from src import data
    from src import train as training

    if round_n != 0:
        raise ValueError(
            "only round 0 (the first training run) exists so far; later rounds come in M9"
        )
    train_rows = data.read_jsonl(config.TRAIN_PATH)
    eval_rows = data.read_jsonl(config.EVAL_PATH)
    if probe:
        train_rows, eval_rows = train_rows[:8], eval_rows[:4]
        epochs = 2 if epochs < 0 else epochs
    rounds = training.read_rounds(config.ROUNDS_PATH)
    if not probe and round_n in [r["round"] for r in rounds]:
        raise ValueError(f"round {round_n} is already recorded in {config.ROUNDS_PATH}")
    if not probe and round_n != len(rounds):
        raise ValueError(
            f"gap in the round numbers: the next round must be {len(rounds)}"
        )
    overrides = {
        "epochs": _sentinel(epochs),
        "per_device_train_batch_size": _sentinel(batch_size),
        "gradient_accumulation_steps": _sentinel(grad_accum),
        "seed": _sentinel(seed),
    }

    # and then, before any GPU time, we print the longest example, the memory estimate and the plan
    tokenizer = data.load_tokenizer(config.MODEL_NAME)
    lengths = data.token_lengths(train_rows + eval_rows, tokenizer)
    data.check_token_limit(lengths, config.MAX_SEQ_LENGTH)
    longest = max(item["total"] for item in lengths)
    settings = training.hyperparameters(**training.clean_overrides(overrides))
    estimate = training.estimate_training_memory_gib(
        longest, settings["per_device_train_batch_size"]
    )["total"]
    steps = training.total_steps(
        len(train_rows),
        settings["per_device_train_batch_size"],
        settings["gradient_accumulation_steps"],
        settings["epochs"],
    )
    print(
        f"{'PROBE' if probe else 'TRAINING'}: {len(train_rows)} train rows, "
        f"{len(eval_rows)} eval rows; longest example {longest} tokens "
        f"(limit {config.MAX_SEQ_LENGTH}); estimated memory {estimate:.1f} GiB on "
        f"{config.GPU_TYPE}; about {steps} optimizer steps over {settings['epochs']} epochs"
    )
    summary_path = config.TRAIN_SUMMARY_PATH
    if summary_path.exists():
        pending = json.loads(summary_path.read_text(encoding="utf-8")).get("pending", 0)
        if pending:
            print(
                f"WARNING: train.jsonl is incomplete, {pending} rows still pending; treat the results as provisional"
            )

    remote = train_on_gpu.remote(train_rows, eval_rows, round_n, probe, overrides)
    record = remote["record"]
    print(f"masking verified on {remote['masking_checked']} real examples")
    for epoch in record["epoch_results"]:
        print(
            f"  epoch {epoch['epoch']}: train loss {epoch['train_loss']}, eval loss "
            f"{epoch['eval_loss']}, accuracy {epoch['accuracy']}, format "
            f"{epoch['format_compliance']}"
        )
    if probe:
        print(
            "probe finished: nothing was saved to rounds.json and no adapter was kept"
        )
        return
    training.record_round(config.ROUNDS_PATH, record)
    print(
        f"round {round_n} recorded in {config.ROUNDS_PATH}; adapter on the Modal volume"
    )
