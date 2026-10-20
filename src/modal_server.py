import importlib.metadata
import json
import subprocess
import time
import urllib.request

import modal

from src import config

IMPORT_EPOCH = time.time()  # in the container: when our module first runs
PORT = 8000
FAST_BOOT = True  # edit by hand between experiments, then make deploy
VERSIONED_PACKAGES = [
    "vllm",
    "torch",
    "transformers",
    "huggingface_hub",
    "flashinfer-python",
]

app = modal.App("trick-question-serving")

# cache volumes mount at the default Hugging Face and vLLM cache folders; adapters mount read-only
volumes = {
    "/root/.cache/huggingface": modal.Volume.from_name(config.HF_CACHE_VOLUME),
    "/root/.cache/vllm": modal.Volume.from_name(config.VLLM_CACHE_VOLUME),
    "/adapters": modal.Volume.from_name(config.ADAPTERS_VOLUME).read_only(),
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


def vllm_command(fast_boot):
    command = [
        "vllm",
        "serve",
        config.MODEL_NAME,
        "--served-model-name",
        config.SERVED_BASE_NAME,
        "--enable-lora",
        "--lora-modules",
        f"{config.ADAPTER_NAME}={config.ADAPTER_PATH}",
        "--max-lora-rank",
        str(config.LORA_R),
        "--max-model-len",
        str(config.EVAL_MAX_SEQ_LENGTH),
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
    ]
    return command + ["--enforce-eager"] if fast_boot else command


def marks_line(marks):
    return f"{config.MARKS_TAG} {json.dumps(marks)}"


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
        print("versions:", versions, "| FAST_BOOT:", FAST_BOOT)
        launch_epoch = time.time()
        self.process = subprocess.Popen(vllm_command(FAST_BOOT))
        wait_until_ready()
        marks = {
            "import_epoch": IMPORT_EPOCH,
            "launch_epoch": launch_epoch,
            "ready_epoch": time.time(),
        }
        print(marks_line(marks), flush=True)

    @modal.exit()
    def stop(self):
        self.process.terminate()
