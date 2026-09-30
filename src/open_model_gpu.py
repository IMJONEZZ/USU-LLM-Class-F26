import modal

app = modal.App("gpu-test")

image = modal.Image.debian_slim().pip_install(
    "torch",
    "transformers",
    "numpy",
    "accelerate",
    "bitsandbytes",
)


@app.function(
    image=image,
    gpu="T4",
)
def run_model():
    import logging

    logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
    import subprocess

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        pipeline,
    )

    # Load tokenizer and model
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name())

    model_id = "Qwen/Qwen2.5-7B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=0,
        quantization_config=quantization_config,
    )

    print(
        subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=False
        ).stdout
    )

    print("Model device:", model.device)
    print("GPU memory allocated:", torch.cuda.memory_allocated() / 1024**3, "GB")
    print("GPU memory reserved:", torch.cuda.memory_reserved() / 1024**3, "GB")
    print("Model dtype:", model.dtype)

    # Create a text generation pipeline
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Generate text
    prompt = "Once upon a time"

    output = generator(prompt, max_new_tokens=50, do_sample=True, temperature=0.7)

    # Print result
    print(output[0]["generated_text"])
