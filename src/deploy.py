import subprocess

import modal
from huggingface_hub import snapshot_download

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.10.2",
        "huggingface_hub[hf_transfer]==0.35.0",
        "flashinfer-python==0.3.1",
        "torch==2.8.0",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # Avoid FlashInfer's sampler JIT failure on this T4 configuration.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
)

# Serve the trained PEFT adapter on top of the same base model used for training.
BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B"
ADAPTER_NAME = "monslogan/A6-llama-3.2-1b-lora"
SERVED_MODEL_NAME = "A6-llama-3.2-1b-lora"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

FAST_BOOT = True

app = modal.App("example-vllm-inference")

N_GPU = 1
MINUTES = 60  # seconds
VLLM_PORT = 8000
ROUTING_REGION = "us-east"
hf_secret = modal.Secret.from_local_environ(["HF_TOKEN"])


@app.server(
    image=vllm_image,
    gpu=f"T4:{N_GPU}",
    scaledown_window=900,  # how long should we stay up with no requests?
    startup_timeout=3 * MINUTES,  # how long should we wait for container start?
    target_concurrency=20,  # how many requests can one replica handle? tune carefully!
    port=VLLM_PORT,
    routing_region=ROUTING_REGION,
    unauthenticated=True,
    secrets=[hf_secret],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
class Server:
    @modal.enter()
    def startup(self) -> None:
        # Download the adapter's current default revision. The Llama base repo
        # is gated, so HF_TOKEN is supplied through the Modal secret.
        adapter_path = snapshot_download(
            repo_id=ADAPTER_NAME,
            token=True,
        )
        cmd = [
            "vllm",
            "serve",
            "--uvicorn-log-level=info",
            BASE_MODEL_NAME,
            "--enable-lora",
            "--max-lora-rank",
            "32",
            "--lora-modules",
            f"{SERVED_MODEL_NAME}={adapter_path}",
            "--max-model-len",
            "4096",
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
        ]

        # enforce-eager disables both Torch compilation and CUDA graph capture
        # default is no-enforce-eager. see the --compilation-config flag for tighter control
        cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

        # assume multiple GPUs are for splitting up large matrix multiplications
        cmd += ["--tensor-parallel-size", str(N_GPU)]

        print(cmd)

        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self) -> None:
        self.process.terminate()
