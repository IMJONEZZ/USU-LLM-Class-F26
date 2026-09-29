import modal

MODEL_ID = "meta-llama/Llama-3.2-1B"

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "transformers[torch]"
)

app = modal.App("llama-3-2-1b-gpu-test")


@app.function(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("huggingface")],
)
def generate_text(prompt: str) -> str:
    from transformers import pipeline

    generator = pipeline(
        "text-generation",
        model=MODEL_ID,
        device=0,
    )
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
