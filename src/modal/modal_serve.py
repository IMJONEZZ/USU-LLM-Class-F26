"""Serve the Star Wars LoRA adapter with vLLM on Modal."""

from pathlib import Path

import modal

APP_NAME = "star-wars-lora-serving"
MODEL_NAME = "meta-llama/Llama-3.2-1B"
ADAPTER_NAME = "star-wars"
MODEL_CACHE_PATH = "/model-cache"
CHECKPOINT_PATH = "/checkpoints/star-wars-llama"
VLLM_CACHE_PATH = "/root/.cache/vllm"
VLLM_PORT = 8000

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.12.0")
    .env(
        {
            "HF_HOME": MODEL_CACHE_PATH,
            "VLLM_CACHE_ROOT": VLLM_CACHE_PATH,
        }
    )
)

model_cache = modal.Volume.from_name("star-wars-model-cache", create_if_missing=True)
checkpoints = modal.Volume.from_name(
    "star-wars-lora-checkpoints", create_if_missing=True
)
vllm_cache = modal.Volume.from_name("star-wars-vllm-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    timeout=20 * 60,
    scaledown_window=60,
    volumes={
        MODEL_CACHE_PATH: model_cache,
        "/checkpoints": checkpoints,
        VLLM_CACHE_PATH: vllm_cache,
    },
    secrets=[modal.Secret.from_name("huggingface")],
)
@modal.web_server(port=VLLM_PORT, startup_timeout=15 * 60)
def serve() -> None:
    """Start vLLM's OpenAI-compatible API server with the saved LoRA adapter."""
    import subprocess

    adapter_config = Path(CHECKPOINT_PATH) / "adapter_config.json"
    if not adapter_config.is_file():
        raise FileNotFoundError(
            f"No LoRA adapter found at {CHECKPOINT_PATH}; "
            f"expected {adapter_config.name}"
        )

    command = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--dtype",
        "half",
        "--max-model-len",
        "2048",
        "--gpu-memory-utilization",
        "0.9",
        "--enable-lora",
        "--max-lora-rank",
        "16",
        "--lora-modules",
        f"{ADAPTER_NAME}={CHECKPOINT_PATH}",
    ]
    subprocess.Popen(command)
