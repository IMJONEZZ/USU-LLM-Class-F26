import modal

app = modal.App("resume-baseline-eval")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "unsloth", "datasets", "pandas==3.0.6", "sentence-transformers", "tqdm"
    )
    .env({"HF_HOME": "/hf_cache"})
    .add_local_dir("src", remote_path="/root/src")
)


@app.function(gpu="T4", image=image, volumes={"/hf_cache": hf_cache}, timeout=3600)
def run_baseline(model_name, n, batch_size, max_seq_length):
    import pandas as pd
    from datasets import load_dataset
    from unsloth import FastLanguageModel  # unsloth must be imported first

    from src.evaluator import evaluate_model, load_eval_subset
    from src.generation import make_generate_fn
    from src.metrics import load_embedding_model
    from src.preprocessing import sample_stratified, split_stratified
    from src.prompts import format_prompt
    from src.utils import GpuMonitor, Timer, peak_memory_allocated_mib

    # and then we rebuild the same seeded splits the local pipeline makes
    resumes = load_dataset("Divyaamith/Kaggle-Resume")["train"].to_pandas()
    sampled_resumes = sample_stratified(resumes)
    _train_resumes, _validation_resumes, test_resumes = split_stratified(
        sampled_resumes
    )
    eval_resumes = load_eval_subset(test_resumes, n=n)
    print("first test-split IDs (compare with local):", list(test_resumes["ID"][:5]))
    print("test split size:", len(test_resumes), "| evaluating:", len(eval_resumes))

    # and then we load the base model in 4-bit, like the QLoRA run will use
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        device_map={"": 0},
    )
    FastLanguageModel.for_inference(model)

    # and then we check how many prompts exceed the context limit we set
    prompts = [format_prompt(text) for text in eval_resumes["Resume_str"]]
    prompt_lengths = pd.Series([len(tokenizer(p)["input_ids"]) for p in prompts])
    print("prompt token lengths:")
    print(prompt_lengths.describe(percentiles=[0.5, 0.9, 0.99]).round(0))
    print(
        f"prompts longer than max_seq_length={max_seq_length}:",
        int((prompt_lengths > max_seq_length).sum()),
    )

    # and then we read a few raw generations by eye before trusting any metric
    generate_fn = make_generate_fn(model, tokenizer)
    sample_generations = generate_fn(prompts[:3])
    for generated, expected in zip(sample_generations, eval_resumes["Category"][:3]):
        print(f"generated={generated!r} | expected={expected!r}")

    # and then we run the full evaluation with the timer and GPU monitor on
    embedding_model = load_embedding_model()
    categories = sorted(resumes["Category"].unique())
    with Timer() as timer, GpuMonitor() as gpu_monitor:
        results = evaluate_model(
            generate_fn, eval_resumes, categories, embedding_model, batch_size
        )

    print("results:", results)
    print(f"eval seconds: {timer.elapsed_seconds:.1f}")
    print("gpu summary:", gpu_monitor.summary)
    print(f"peak torch memory allocated (MiB): {peak_memory_allocated_mib():.0f}")
    hf_cache.commit()
    return results


@app.local_entrypoint()
def main(
    model_name: str = "unsloth/Llama-3.2-1B",
    n: int = 100,
    batch_size: int = 8,
    max_seq_length: int = 4096,
):
    run_baseline.remote(model_name, n, batch_size, max_seq_length)
