"""Question 2: run one rung of the model-size ladder on a Modal T4.

Usage:
    modal run assignment_4/gpu_ladder.py --model-id Qwen/Qwen2.5-1.5B-Instruct --precision fp16
    modal run assignment_4/gpu_ladder.py --model-id Qwen/Qwen2.5-14B-Instruct --precision 4bit

"Fits" is strict: the whole model is placed on GPU 0. Nothing may spill to CPU
memory, so a model that is too large raises an out-of-memory error.
"""

import json
import subprocess
import time
from pathlib import Path

import modal

GPU = "T4"
PROMPT = (
    "Explain in three sentences why GPU memory limits the size of a language model."
)
MAX_NEW_TOKENS = 120
BYTES_PER_PARAM = {"fp16": 2.0, "4bit": 0.6}

app = modal.App("a4-gpu-ladder")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch", "transformers", "accelerate", "bitsandbytes", "huggingface_hub"
    )
    .env({"HF_HOME": "/cache/huggingface"})
)

# Keeps downloaded weights between runs, so a model is only downloaded once.
cache = modal.Volume.from_name("a4-hf-cache", create_if_missing=True)


def nvidia_smi() -> str:
    return subprocess.run(
        ["nvidia-smi"], capture_output=True, text=True, check=False
    ).stdout


@app.function(
    gpu=GPU, image=image, volumes={"/cache": cache}, memory=32768, timeout=3600
)
def run_model(model_id: str, precision: str) -> dict:
    import torch
    from huggingface_hub import model_info
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    params = model_info(model_id).safetensors.total
    result = {
        "model_id": model_id,
        "precision": precision,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_gib": round(
            torch.cuda.get_device_properties(0).total_memory / 2**30, 2
        ),
        "params_billion": round(params / 1e9, 2),
        "estimated_gib": round(params * BYTES_PER_PARAM[precision] / 2**30, 2),
    }

    if precision == "4bit":
        load_kwargs = {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        }
    else:
        load_kwargs = {"dtype": torch.float16}

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        start = time.perf_counter()
        # device_map={"": 0} puts every layer on GPU 0; "auto" could offload to CPU.
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map={"": 0}, **load_kwargs
        )
        result["load_seconds"] = round(time.perf_counter() - start, 1)

        messages = [{"role": "user", "content": PROMPT}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to("cuda")
        prompt_tokens = inputs["input_ids"].shape[1]

        start = time.perf_counter()
        output = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )
        elapsed = time.perf_counter() - start
        new_tokens = output.shape[1] - prompt_tokens

        result["status"] = "fits"
        result["tokens_per_second"] = round(new_tokens / elapsed, 2)
        result["new_tokens"] = new_tokens
        result["sample"] = tokenizer.decode(
            output[0][prompt_tokens:], skip_special_tokens=True
        )
    except torch.OutOfMemoryError as err:
        result["status"] = "does not fit"
        result["error"] = str(err).split("\n")[0]

    result["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated(0) / 2**30, 2)
    result["nvidia_smi"] = nvidia_smi()
    cache.commit()
    return result


@app.local_entrypoint()
def main(model_id: str, precision: str = "fp16"):
    result = run_model.remote(model_id, precision)
    smi = result.pop("nvidia_smi")
    print(smi)
    print(json.dumps(result, indent=2))

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    name = f"{model_id.split('/')[-1]}_{precision}"
    (out_dir / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    (out_dir / f"{name}_nvidia_smi.txt").write_text(smi)
