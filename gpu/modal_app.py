"""Runs a text generation model on a rented NVIDIA GPU through Modal.

Loads a model onto the GPU, prints nvidia-smi while the weights are resident,
generates some text, and reports how much GPU memory it actually used. The
point is finding where a model stops fitting on a 16 GB T4.

Loading pins everything to GPU 0 with device_map={"": 0} rather than "auto".
"auto" will quietly spill layers onto the CPU when it runs out of room, and a
model running half on the CPU hasn't fit, it just hasn't crashed. Pinning makes
that show up as an error instead of a slow success.

Run it from the repo root:

    modal run gpu/modal_app.py                      # the whole ladder
    modal run gpu/modal_app.py --model-id MODEL     # one model
"""

import modal

GPU_TYPE = "T4"
CACHE_DIR = "/cache"

# This is the container image config the assignment asks me to submit. Modal
# builds it once and reuses it, so the pip install only happens on the first run.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.49.0",
        "accelerate==1.4.0",
        "bitsandbytes==0.45.3",
        "huggingface_hub[hf_transfer]==0.29.1",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": CACHE_DIR})
)

app = modal.App("hw4-gpu", image=image)

# Weights get cached in a volume so climbing the ladder doesn't re-download
# several GB every time.
hf_cache = modal.Volume.from_name("hw4-hf-cache", create_if_missing=True)

# The ladder. The Llama repos are gated so this needs the HF token secret. I
# only got approved for 1B and 3B, so the top of the ladder is Qwen instead,
# which is ungated and has a 7B that lands just over what a T4 can hold.
LADDER = [
    ("meta-llama/Llama-3.2-1B", "fp16"),
    ("meta-llama/Llama-3.2-3B", "fp16"),
    ("Qwen/Qwen2.5-7B", "fp16"),
    ("Qwen/Qwen2.5-7B", "8bit"),
    ("Qwen/Qwen2.5-7B", "4bit"),
]


def _nvidia_smi():
    """Grab nvidia-smi output so it can go in the writeup."""
    import subprocess

    full = subprocess.run(
        ["nvidia-smi"], capture_output=True, text=True, check=False
    ).stdout
    # Containers often can't see host PIDs, so ask for compute apps directly as
    # well. Between the two, one of them should show the process using memory.
    apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return full, apps


@app.function(
    gpu=GPU_TYPE,
    timeout=1800,
    volumes={CACHE_DIR: hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_model(model_id: str, mode: str = "fp16", max_new_tokens: int = 50):
    """Load one model at one precision and try to generate with it."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    device_name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\n=== {model_id}  [{mode}] ===")
    print(f"gpu: {device_name}  ({total_gb:.1f} GB)")

    load_args = {"device_map": {"": 0}}
    if mode == "fp16":
        load_args["torch_dtype"] = torch.float16
    elif mode == "8bit":
        load_args["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif mode == "4bit":
        load_args["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        raise ValueError(f"unknown mode {mode}")

    torch.cuda.reset_peak_memory_stats()
    result = {"model_id": model_id, "mode": mode, "gpu": device_name}

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_args)
    except Exception as exc:  # noqa: BLE001 - too big shows up as several types
        # This is the branch that catches a model that doesn't fit. Pinning to
        # GPU 0 means an oversized model raises here instead of silently
        # offloading, which is the whole reason for the pin. Out of memory comes
        # back as OutOfMemoryError, RuntimeError or ValueError depending on where
        # it gives up, so this catches all of them and reports what it was.
        print(f"FAILED to load: {type(exc).__name__}: {str(exc)[:400]}")
        result.update({"loaded": False, "error": f"{type(exc).__name__}: {exc}"})
        return result

    params = sum(p.numel() for p in model.parameters())
    after_load = torch.cuda.memory_allocated() / 1024**3
    print(f"parameters: {params / 1e9:.3f}B")
    print(f"memory after load: {after_load:.2f} GB")

    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt").to(0)

    try:
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7
        )
    except Exception as exc:  # noqa: BLE001 - same reason as the load above
        print(f"FAILED to generate: {type(exc).__name__}: {str(exc)[:400]}")
        result.update({"loaded": True, "generated": False, "error": str(exc)})
        return result

    # nvidia-smi goes here, while the weights are still resident and generation
    # has just run, so the memory column isn't zero when I screenshot it.
    full, apps = _nvidia_smi()
    print("\n--- nvidia-smi ---")
    print(full)
    print("--- compute apps ---")
    print(apps)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\npeak memory: {peak:.2f} GB of {total_gb:.1f} GB")
    print(f"output: {text}")

    result.update(
        {
            "loaded": True,
            "generated": True,
            "params_b": round(params / 1e9, 3),
            "memory_after_load_gb": round(after_load, 2),
            "peak_memory_gb": round(peak, 2),
            "total_memory_gb": round(total_gb, 1),
            "output": text,
        }
    )
    return result


@app.local_entrypoint()
def main(model_id: str = "", mode: str = "fp16"):
    """Run one model, or the whole ladder if no model is given."""
    runs = [(model_id, mode)] if model_id else LADDER

    results = []
    for mid, m in runs:
        results.append(run_model.remote(mid, m))

    print("\n\n=== summary ===")
    for r in results:
        if not r.get("loaded"):
            print(f"  {r['model_id']:32} {r['mode']:5}  DID NOT FIT")
        elif not r.get("generated"):
            print(f"  {r['model_id']:32} {r['mode']:5}  loaded but failed to generate")
        else:
            print(
                f"  {r['model_id']:32} {r['mode']:5}  "
                f"{r['params_b']:6.3f}B  peak {r['peak_memory_gb']:5.2f} GB  OK"
            )
