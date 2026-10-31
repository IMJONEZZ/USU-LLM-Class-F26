import importlib.metadata
import subprocess
import time
import urllib.request

import modal

from src import config

PORT = 8000
VERSIONED_PACKAGES = [
    "vllm",
    "torch",
    "transformers",
    "huggingface_hub",
    "flashinfer-python",
]

app = modal.App("trick-question-serving")

# cache volumes mount at the default Hugging Face and vLLM cache folders
volumes = {
    "/root/.cache/huggingface": modal.Volume.from_name(config.HF_CACHE_VOLUME),
    "/root/.cache/vllm": modal.Volume.from_name(config.VLLM_CACHE_VOLUME),
}

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.10.2",
        "torch==2.8.0",
        "flashinfer-python==0.3.1",
        "huggingface_hub==0.35.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_dir("src", remote_path="/root/src")
)


def vllm_command():
    return [
        "vllm",
        "serve",
        config.MODEL_NAME,
        "--served-model-name",
        config.SERVED_BASE_NAME,
        "--max-model-len",
        str(config.EVAL_MAX_SEQ_LENGTH),
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--enforce-eager",  # skips CUDA graph capture, so the server boots faster
    ]


def wait_until_ready():
    while True:
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2)
            return
        except OSError:
            time.sleep(1)


@app.server(
    image=image,
    gpu=config.GPU_TYPE,
    volumes=volumes,
    scaledown_window=config.SCALEDOWN_WINDOW,
    startup_timeout=10 * 60,
    port=PORT,
)
class Server:
    @modal.enter()
    def start(self):
        versions = {
            name: importlib.metadata.version(name) for name in VERSIONED_PACKAGES
        }
        print("versions:", versions)
        self.process = subprocess.Popen(vllm_command())
        wait_until_ready()

    @modal.exit()
    def stop(self):
        self.process.terminate()
