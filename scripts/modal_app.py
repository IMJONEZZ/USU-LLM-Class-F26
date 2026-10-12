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
