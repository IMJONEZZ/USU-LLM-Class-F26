import json
from pathlib import Path

import modal

app = modal.App("resume-baseline-eval")
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
    timeout=3600,
)
def run_baseline(model_name, batch_size, max_seq_length, adapter_path):
    from unsloth import FastLanguageModel  # isort: skip  (unsloth must be first)

    from importlib import metadata

    import pandas as pd
    from datasets import load_dataset
    from tqdm import tqdm

    from src.evaluator import evaluate_model, load_eval_subset
    from src.generation import make_generate_fn
    from src.metrics import load_embedding_model
    from src.preprocessing import make_splits
    from src.prompts import format_example, format_prompt
    from src.utils import GpuMonitor, Timer, peak_memory_allocated_mib

    # and then we record the exact package versions this run used
    versions = {
        package: metadata.version(package)
        for package in [
            "torch",
            "torchao",
            "unsloth",
            "transformers",
            "sentence-transformers",
            "datasets",
            "pandas",
            "numpy",
        ]
    }
    print("versions:", versions)

    # and then we rebuild the same cleaned, seeded splits the other scripts use
    resumes = load_dataset("Divyaamith/Kaggle-Resume")["train"].to_pandas()
    train_resumes, validation_resumes, test_resumes = make_splits(resumes)
    eval_resumes = load_eval_subset(test_resumes, n=len(test_resumes))
    print("first test-split IDs (compare with local):", list(test_resumes["ID"][:5]))
    print("test split size:", len(test_resumes), "| evaluating:", len(eval_resumes))

    # and then we load the model in 4-bit (the saved adapter if one is given)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path or model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        device_map={"": 0},
    )
    FastLanguageModel.for_inference(model)
    model.generation_config.max_length = None

    # and then we measure token lengths of full training strings in every split
    categories = sorted(resumes["Category"].unique())
    for split_name, split_resumes in [
        ("train", train_resumes),
        ("validation", validation_resumes),
        ("test", test_resumes),
    ]:
        split_lengths = pd.Series(
            [
                len(tokenizer(format_example(text, category, categories))["input_ids"])
                for text, category in tqdm(
                    zip(split_resumes["Resume_str"], split_resumes["Category"]),
                    total=len(split_resumes),
                )
            ]
        )
        print(f"{split_name} token lengths (prompt + label):")
        print(split_lengths.describe(percentiles=[0.5, 0.9, 0.99]).round(0))
        print(
            f"{split_name} examples longer than max_seq_length={max_seq_length}:",
            int((split_lengths > max_seq_length).sum()),
        )

    # and then we read a few raw generations by eye before trusting any metric
    prompts = [format_prompt(text, categories) for text in eval_resumes["Resume_str"]]
    generate_fn = make_generate_fn(model, tokenizer)
    sample_generations = generate_fn(prompts[:3])
    for generated, expected in zip(sample_generations, eval_resumes["Category"][:3]):
        print(f"generated={generated!r} | expected={expected!r}")

    # and then we run the full evaluation with the timer and GPU monitor on
    embedding_model = load_embedding_model()
    with Timer() as timer, GpuMonitor() as gpu_monitor:
        results = evaluate_model(
            generate_fn, eval_resumes, categories, embedding_model, batch_size
        )

    print("results:", results)
    print(f"eval seconds: {timer.elapsed_seconds:.1f}")
    print("gpu summary:", gpu_monitor.summary)
    print(f"peak torch memory allocated (MiB): {peak_memory_allocated_mib():.0f}")
    hf_cache.commit()
    return {
        "model_name": model_name,
        "adapter_path": adapter_path,
        "test_size": len(test_resumes),
        "batch_size": batch_size,
        "max_seq_length": max_seq_length,
        "versions": versions,
        "metrics": results,
        "eval_seconds": timer.elapsed_seconds,
        "gpu_summary": gpu_monitor.summary,
        "peak_torch_memory_allocated_mib": peak_memory_allocated_mib(),
    }


@app.local_entrypoint()
def main(
    model_name: str = "unsloth/Llama-3.2-1B",
    batch_size: int = 8,
    max_seq_length: int = 4096,
    adapter_path: str = "",
    output_name: str = "baseline_eval",
):
    eval_results = run_baseline.remote(
        model_name, batch_size, max_seq_length, adapter_path
    )
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{output_name}.json").write_text(json.dumps(eval_results, indent=2))
