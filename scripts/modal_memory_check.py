import modal

app = modal.App("resume-memory-check")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

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


@app.function(gpu="T4", image=image, volumes={"/hf_cache": hf_cache}, timeout=1800)
def check_worst_case_memory(batch_sizes):

    import torch

    from src.config import MAX_SEQ_LENGTH
    from src.trainer import load_model_for_training
    from src.utils import GpuMonitor, peak_memory_allocated_mib

    # and then we load the model exactly the way training will
    model, _ = load_model_for_training()
    model.train()
    vocab_size = model.config.vocab_size
    print(f"max_seq_length={MAX_SEQ_LENGTH}, vocab_size={vocab_size}")

    # and then we run one forward and backward pass on a full-length batch
    outcomes = []
    for batch_size in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        input_ids = torch.randint(
            0, vocab_size, (batch_size, MAX_SEQ_LENGTH), device="cuda"
        )
        try:
            with GpuMonitor(interval=1.0) as gpu_monitor:
                loss = model(input_ids=input_ids, labels=input_ids).loss
                loss.backward()
            outcome = {
                "batch_size": batch_size,
                "fits": True,
                "peak_torch_memory_allocated_mib": peak_memory_allocated_mib(),
                "gpu_summary": gpu_monitor.summary,
            }
        except torch.cuda.OutOfMemoryError:
            outcome = {"batch_size": batch_size, "fits": False}
        model.zero_grad(set_to_none=True)
        print(outcome)
        outcomes.append(outcome)
    return outcomes


@app.local_entrypoint()
def main():
    outcomes = check_worst_case_memory.remote([1, 2])
    for outcome in outcomes:
        print(outcome)
