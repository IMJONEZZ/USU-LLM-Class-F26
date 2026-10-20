"""Train and save the Star Wars LoRA adapter on Modal."""

import modal

APP_NAME = "star-wars-lora-training"
MODEL_NAME = "meta-llama/Llama-3.2-1B"
MODEL_CACHE_PATH = "/model-cache"
CHECKPOINT_PATH = "/checkpoints/star-wars-llama"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "unsloth",
        "transformers>=4,<5",
        "datasets>=3.0.0",
        "evaluate>=0.4.6",
        "peft>=0.17.0",
        "accelerate>=0.26.0",
        "rouge-score>=0.1.2",
    )
    .env({"HF_HOME": MODEL_CACHE_PATH})
    .add_local_python_source("src")
)

model_cache = modal.Volume.from_name("star-wars-model-cache", create_if_missing=True)
checkpoints = modal.Volume.from_name(
    "star-wars-lora-checkpoints", create_if_missing=True
)


@app.function(
    image=image,
    gpu="T4",
    timeout=4 * 60 * 60,
    volumes={
        MODEL_CACHE_PATH: model_cache,
        "/checkpoints": checkpoints,
    },
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_experiment() -> dict[str, object]:
    """Train the LoRA adapter and save it to the checkpoint Volume."""
    from src.trainer import run_training

    training_metrics = run_training(
        model_name=MODEL_NAME,
        output_dir=CHECKPOINT_PATH,
    )
    checkpoints.commit()
    return {"training": training_metrics}


@app.local_entrypoint()
def main() -> None:
    scores = run_experiment.remote()
    print(scores)
