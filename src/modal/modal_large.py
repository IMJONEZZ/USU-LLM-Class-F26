import modal

MODEL_ID = "Qwen/Qwen3-8B"

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "transformers[torch]",
    "accelerate",
    "bitsandbytes",
)

app = modal.App("qwen3-8b-gpu-test")


@app.function(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("huggingface")],
)
def generate_text(prompt: str) -> str:
    import subprocess

    import torch
    from transformers import BitsAndBytesConfig, pipeline

    generator = pipeline(
        "text-generation",
        model=MODEL_ID,
        device_map="auto",
        model_kwargs={
            "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
            "dtype": torch.float16,
        },
    )
    subprocess.run(["nvidia-smi"], check=True)
    output = generator(
        prompt,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
    )
    return output[0]["generated_text"]


@app.local_entrypoint()
def main(prompt: str = "A long, long time ago") -> None:
    print(generate_text.remote(prompt))
