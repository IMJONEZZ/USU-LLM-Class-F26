"""Assignment 7: serve Llama 3.2 1B Instruct plus my Assignment 6 LoRA adapter.

`test_lora` is a one-off check that vLLM can load the adapter on a T4. `serve`
is the deployed OpenAI-compatible server (`modal deploy`), protected by an API
key from the `a7-vllm-api-key` Modal secret.
"""

from __future__ import annotations

import os

import modal

APP_NAME = "assignment-7-vllm"
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
BASE_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
VLLM_VERSION = os.environ.get("A7_VLLM_VERSION", "0.32.0")
ADAPTER_PATH = "/adapter/run16"
ADAPTER_NAME = "a6"
PORT = 8000
CACHE_PATH = "/model_cache"

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub")
    .env({"HF_HOME": CACHE_PATH})
)
model_cache = modal.Volume.from_name("a7-model-cache", create_if_missing=True)
adapter = modal.Volume.from_name("a7-adapter")


@app.function(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CACHE_PATH: model_cache, "/adapter": adapter},
    timeout=1800,
)
def test_lora(question: str) -> dict:
    """Load the base model with the adapter and answer one question both ways."""
    import time

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    start = time.time()
    llm = LLM(
        model=BASE_MODEL,
        revision=BASE_REVISION,
        dtype="half",
        enable_lora=True,
        max_lora_rank=16,
        max_model_len=2048,
    )
    load_seconds = time.time() - start
    params = SamplingParams(temperature=0, max_tokens=512)
    messages = [[{"role": "user", "content": question}]]
    base = llm.chat(messages, params)[0].outputs[0].text
    tuned = (
        llm.chat(messages, params, lora_request=LoRARequest("a6", 1, ADAPTER_PATH))[0]
        .outputs[0]
        .text
    )
    return {
        "vllm_version": VLLM_VERSION,
        "load_seconds": round(load_seconds, 1),
        "base": base,
        "adapter": tuned,
    }


@app.function(
    image=image,
    gpu="T4",
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("a7-vllm-api-key"),
    ],
    volumes={CACHE_PATH: model_cache, "/adapter": adapter},
    scaledown_window=300,
    timeout=3600,
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=PORT, startup_timeout=600)
def serve() -> None:
    """Start an OpenAI-compatible vLLM server for the base model and adapter."""
    import subprocess

    subprocess.Popen(
        [
            "vllm",
            "serve",
            BASE_MODEL,
            "--revision",
            BASE_REVISION,
            "--dtype",
            "half",
            "--max-model-len",
            "2048",
            "--enable-lora",
            "--max-lora-rank",
            "16",
            "--lora-modules",
            f"{ADAPTER_NAME}={ADAPTER_PATH}",
            "--api-key",
            os.environ["VLLM_API_KEY"],
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
        ]
    )


@app.local_entrypoint()
def main(question: str = "How many Ls are in the word parallel?") -> None:
    result = test_lora.remote(question)
    for key, value in result.items():
        print(f"{key}: {value}")
