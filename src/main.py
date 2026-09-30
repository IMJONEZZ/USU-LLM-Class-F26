import modal

from src.generation import GPU_TYPE, app, build_image, generate

MODEL_CHECKPOINTS = [
    "unsloth/Llama-3.2-1B",
    "unsloth/Llama-3.2-3B",
    "unsloth/Meta-Llama-3.1-8B",
]
BIT_WIDTHS = ["16bit", "4bit"]


@app.function(
    gpu=GPU_TYPE,
    image=build_image().add_local_python_source("src"),
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
)
def main() -> None:
    import gc

    import torch

    results = {}
    for model_name in MODEL_CHECKPOINTS:
        for bit_width in BIT_WIDTHS:
            fits, peak_mib, peak_percent = generate.local(model_name, bit_width)
            results[(model_name, bit_width)] = (fits, peak_mib, peak_percent)

            gc.collect()
            torch.cuda.empty_cache()

    for (model_name, bit_width), (fits, peak_mib, peak_percent) in results.items():
        status = "FIT" if fits else "NO FIT"
        print(
            f"[scale-up] {model_name} ({bit_width}): {status}, "
            f"peak GPU usage: {peak_mib:.0f} MiB ({peak_percent:.1f}%)"
        )
