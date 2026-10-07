"""Runs the fine tune on a rented T4 through Modal.

Scores perplexity on the held out half, trains, then scores the same chunks
again so the before and after numbers are comparable. The actual logic lives in
`src/trainer.py`; this file is just the GPU wiring.

    modal run gpu/train_app.py
"""

import modal

GPU_TYPE = "T4"
CACHE_DIR = "/cache"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "unsloth==2025.3.19",
        "unsloth_zoo==2025.3.17",
        "transformers==4.49.0",
        "trl==0.15.2",
        "peft==0.14.0",
        "accelerate==1.4.0",
        "bitsandbytes==0.45.3",
        "datasets==3.3.2",
        "huggingface_hub[hf_transfer]==0.29.1",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": CACHE_DIR})
    .add_local_python_source("src")
    .add_local_file("SW_EpisodeIV_VI.json", "/root/SW_EpisodeIV_VI.json")
)

app = modal.App("hw5-train", image=image)
hf_cache = modal.Volume.from_name("hw5-hf-cache", create_if_missing=True)


@app.function(
    gpu=GPU_TYPE,
    timeout=3600,
    volumes={CACHE_DIR: hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_training(max_steps: int = 60):
    import subprocess

    import torch

    from src.trainer import (
        attach_lora,
        build_splits,
        finetune,
        load_base_model,
        perplexity,
        summarize,
    )

    print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(
        f"total vram: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )

    model, tokenizer = load_base_model()
    train_ids, val_ids = build_splits(tokenizer, path="/root/SW_EpisodeIV_VI.json")
    print(f"train tokens {len(train_ids)}, val tokens {len(val_ids)}")

    # Baseline first, before anything touches the weights.
    before = perplexity(model, val_ids, device="cuda")
    print(f"BEFORE  loss {before['loss']:.4f}  ppl {before['perplexity']:.3f}")

    prompts = ["Luke, ", "THREEPIO: ", "I have a bad feeling about "]

    def sample(m, prompt):
        ids = tokenizer(prompt, return_tensors="pt").to("cuda")
        out = m.generate(
            **ids, max_new_tokens=50, do_sample=True, temperature=0.7, top_p=0.9
        )
        return tokenizer.decode(out[0], skip_special_tokens=True)

    samples_before = {p: sample(model, p) for p in prompts}
    print("\n--- samples BEFORE ---")
    for p, t in samples_before.items():
        print(f"  [{p}] {t!r}")

    torch.cuda.reset_peak_memory_stats()
    model = attach_lora(model)
    stats = finetune(model, tokenizer, train_ids, max_steps=max_steps)
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
    print("\n--- nvidia-smi during training ---")
    print(smi.stdout)

    after = perplexity(model, val_ids, device="cuda")
    print(f"AFTER   loss {after['loss']:.4f}  ppl {after['perplexity']:.3f}")

    summary = summarize(before, after, dict(stats.metrics))
    summary["peak_vram_gb"] = round(peak_gb, 2)
    summary["gpu"] = torch.cuda.get_device_name(0)
    summary["max_steps"] = max_steps
    summary["train_tokens"] = len(train_ids)
    summary["val_tokens"] = len(val_ids)

    samples_after = {p: sample(model, p) for p in prompts}
    print("\n--- samples AFTER ---")
    for p, t in samples_after.items():
        print(f"  [{p}] {t!r}")
    summary["samples_before"] = samples_before
    summary["samples_after"] = samples_after

    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary


@app.local_entrypoint()
def main(max_steps: int = 60):
    run_training.remote(max_steps)
