"""Compare the base and LoRA Star Wars models on Modal."""

from pathlib import Path

import modal

APP_NAME = "star-wars-lora-evaluation"
MODEL_NAME = "meta-llama/Llama-3.2-1B"
MODEL_CACHE_PATH = "/model-cache"
CHECKPOINT_PATH = "/checkpoints/star-wars-llama"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch",
        "transformers>=4.45.2,<5",
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
def evaluate_models() -> dict[str, dict[str, float]]:
    """Evaluate the base and saved LoRA models on the same Episode VI split."""
    import torch

    from src.evaluator import run_evaluation

    adapter_config = Path(CHECKPOINT_PATH) / "adapter_config.json"
    if not adapter_config.is_file():
        raise FileNotFoundError(
            f"No LoRA adapter found at {CHECKPOINT_PATH}; expected {adapter_config.name}"
        )

    base_scores = run_evaluation(model_name=MODEL_NAME)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    adapter_scores = run_evaluation(
        model_name=MODEL_NAME,
        adapter_path=CHECKPOINT_PATH,
    )
    return {"base_model_rouge": base_scores, "lora_model_rouge": adapter_scores}


@app.local_entrypoint()
def main() -> None:
    scores = evaluate_models.remote()
    print(scores)
