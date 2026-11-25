from pathlib import Path

import modal

volume = modal.Volume.from_name("base-model-store", create_if_missing=True)

vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


MODEL_NAME = "unsloth/Llama-3.2-1B"
MODEL_DIR = Path("/models")

download_image = (
    modal.Image.debian_slim()
    .uv_pip_install("huggingface_hub")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  # enable fast data transfer
)

inference_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install(
        "vllm==0.10.2",
        "torch==2.8.0",
        "flashinfer-python==0.3.1",
        "huggingface_hub==0.35.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)
app = modal.App()


@app.function(
    volumes={MODEL_DIR: volume},
    image=download_image,
    secrets=[modal.Secret.from_dotenv()],
)  # .as_posix() converts a path objevt into a string
def download_model():
    import os

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=MODEL_DIR / MODEL_NAME,
        token=os.environ["HF_TOKEN"],
    )
    print(f"Model downloaded to {MODEL_DIR / MODEL_NAME}")
    volume.commit()
    print("Model commited!")


@app.function(volumes={MODEL_DIR: volume}, image=inference_image)
def inference():
    from transformers import pipeline

    pipe = pipeline("text-generation", model=MODEL_DIR / MODEL_NAME)

    returned_text = pipe("The secret to baking a good cake is ", max_length=50)

    print(f"Model loaded! {returned_text}")
