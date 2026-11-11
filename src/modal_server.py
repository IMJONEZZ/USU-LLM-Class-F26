import importlib.metadata
import subprocess
import time
import urllib.request

import modal

from src import config

PORT = 8000
MIB = 2**20
VERSIONED_PACKAGES = ["vllm", "torch", "transformers"]

app = modal.App("ifeval-vllm")

hf_volume = modal.Volume.from_name(config.HF_CACHE_VOLUME, create_if_missing=True)

# pins from Modal's current vLLM example (modal.com/docs/examples/vllm_inference); replace them with what the first successful run prints
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "transformers==5.8.1")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_dir("src", remote_path="/root/src")
)


def kv_cache_mib(layers, kv_heads, head_dim, tokens, bytes_per_value=2):
    """Memory for the attention cache: one key and one value vector per layer, kv head and token."""
    return 2 * layers * kv_heads * head_dim * bytes_per_value * tokens / MIB


def vllm_command():
    """The `vllm serve` command line; every value comes from config."""
    return [
        "vllm",
        "serve",
        config.MODEL_NAME,
        "--served-model-name",
        config.SERVED_NAME,
        "--dtype",
        config.DTYPE,
        "--max-model-len",
        str(config.MAX_MODEL_LEN),
        "--generation-config",
        "vllm",  # ignore the model's own sampling defaults; only the client's settings apply
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
    ]


def wait_until_ready():  # pragma: no cover - runs inside the container
    while True:
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2)
            return
        except OSError:
            time.sleep(1)


@app.server(
    image=image,
    gpu=config.GPU_TYPE,
    volumes={"/root/.cache/huggingface": hf_volume},
    scaledown_window=config.SCALEDOWN_WINDOW,
    startup_timeout=10 * 60,
    port=PORT,
    # one T4 only: a second container would bill twice and could change the GPU the pass rates were measured on
    max_containers=1,
    target_concurrency=100,
)
class Server:
    @modal.enter()
    def start(self):  # pragma: no cover - needs a GPU machine
        versions = {n: importlib.metadata.version(n) for n in VERSIONED_PACKAGES}
        print("versions:", versions, flush=True)
        self.process = subprocess.Popen(vllm_command())
        wait_until_ready()

    @modal.exit()
    def stop(self):  # pragma: no cover - needs a GPU machine
        self.process.terminate()
