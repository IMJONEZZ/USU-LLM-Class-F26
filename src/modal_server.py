from dotenv import load_dotenv
load_dotenv()

import modal
from pathlib import Path

volume = modal.Volume.from_name("base-model-store", create_if_missing=True)

MODEL_NAME = "unsloth/Llama-3.2-1B"
MODEL_DIR = Path("/models")


download_image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"}) # enable fast data transfer

)
app = modal.App()


@app.function(volumes={MODEL_DIR:volume}, image=download_image ) # .as_posix() converts a path objevt into a string
def download_model():
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=MODEL_NAME, local_dir=MODEL_DIR / MODEL_NAME)
    print(f'Model downloaded to {MODEL_DIR / MODEL_NAME}')

@app.function(volumes={MODEL_DIR:volume})
def inference(prompt, model_id):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR / MODEL_NAME)
    print("Model uploaded!")