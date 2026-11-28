import os
from random import randint

import modal

from src.config import (
    GPU,
    MODEL_DIR,
    MODEL_NAME,
    ROUTING_REGION,
    SCALEDOWN_WINDOW,
    STARTUP_TIMEOUT,
    VLLM_PORT,
)
from src.schemas import Inputs

model_volume = modal.Volume.from_name("base-model-store", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


download_image = (
    modal.Image.debian_slim()
    .uv_pip_install("huggingface_hub")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  # enable fast data transfer
    .add_local_dir("src", remote_path="/root/src")
)

inference_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install(
        "vllm==0.10.2",
        "torch==2.8.0",
        "flashinfer-python==0.3.1",
        "huggingface_hub==0.35.0",
        "fastapi[standard]",
        "pydantic",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_dir("src", remote_path="/root/src")
)
app = modal.App("vllm-inference")


volumes = {MODEL_DIR: model_volume, "/root/.cache/vllm": vllm_cache_vol}


@app.function(
    volumes={MODEL_DIR: model_volume},
    image=download_image,
    secrets=[modal.Secret.from_dotenv()],
)  # .as_posix() converts a path object into a string
def download_model():
    from huggingface_hub import snapshot_download

    if (MODEL_DIR / MODEL_NAME / "model.safetensors").exists:
        print(f"Model already on the volume at {MODEL_DIR / MODEL_NAME}")
        return
    snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=MODEL_DIR / MODEL_NAME,
        token=os.environ["HF_TOKEN"],
    )
    print(f"Model downloaded to {MODEL_DIR / MODEL_NAME}")
    model_volume.commit()
    print("Model commited!")


@app.server(
    image=inference_image,
    gpu=GPU,
    scaledown_window=SCALEDOWN_WINDOW,
    startup_timeout=STARTUP_TIMEOUT,
    volumes=volumes,
    port=VLLM_PORT,
    routing_region=ROUTING_REGION,
    target_concurrency=2,
    unauthenticated=False,
)
class Server:
    @modal.enter()
    def start(self):
        import subprocess

        cmd = [
            "vllm",
            "serve",
            str(MODEL_DIR / MODEL_NAME),
            "--served-model-name",
            "llm",
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--async-scheduling",
            "--enforce-eager",  # help with the cold starts
        ]
        print(*cmd)
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self.process.terminate()


@app.function(image=inference_image)
@modal.fastapi_endpoint(method="POST", requires_proxy_auth=True, docs=True)
def get_idx(inputs: Inputs) -> dict:
    # from prompts import create_prompts
    # from stats import select_best

    # prompts = create_prompts(inputs= inputs)
    # probs = send_to_modal(prompts = prompts)
    # random_int = select_best(probs = probs)

    random_int = randint(0, len(inputs.candidate_questions) - 1)

    return {"best_question": random_int}
