"""The same model as assignment 9 with the grammar taken off.

This exists to be the control. Assignment 9 put `html.gbnf` on the server with
`--grammar-file`, so every response was HTML whether the caller wanted it or
not. Here the sampler is left alone, which means the only thing standing
between a prompt and a page full of markdown is how well the prompt is
written. That's the whole point of assignment 10, and it's also the comparison
the feedback on my last PR asked for.

Everything else is deliberately identical. Same repo, same quant, same T4, and
the same `hw9-model-cache` volume, so the 4.7GB is already sitting there and a
cold start doesn't re-download it. If I changed the model as well I couldn't
say whether a difference came from the prompt or from the hardware.

    modal deploy gpu/serve_plain_app.py
"""

import subprocess

import modal

CACHE_DIR = "/cache"
PORT = 8080
SCALEDOWN_SECONDS = 60
GPU_TYPE = "T4"

MODEL_REPO = "bartowski/Qwen2.5-7B-Instruct-GGUF"
MODEL_QUANT = "Q4_K_M"

# DSPy fans its evaluations out across threads, and llama-server answers one
# request per slot. `--parallel` buys the slots, and the context size is split
# between them, so 16384 over four slots leaves each one the 4096 assignment 9
# ran with. Setting the slots without raising the context would have quietly
# given every request a quarter of the room.
PARALLEL_SLOTS = 4
CONTEXT_PER_SLOT = 4096

LLAMA_CPP_VERSION = "b11636"
LLAMA_CPP_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/"
    f"{LLAMA_CPP_VERSION}/llama-{LLAMA_CPP_VERSION}-bin-ubuntu-cuda-12.8-x64.tar.gz"
)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-runtime-ubuntu24.04", add_python="3.11"
    )
    .apt_install("curl", "libgomp1", "libcurl4-openssl-dev")
    .run_commands(
        "mkdir -p /opt/llama",
        f"curl -fsSL {LLAMA_CPP_URL} -o /tmp/llama.tar.gz",
        "tar -xzf /tmp/llama.tar.gz -C /opt/llama --strip-components=1",
        "rm /tmp/llama.tar.gz",
        "/opt/llama/llama-server --version",
    )
    .env({"LD_LIBRARY_PATH": "/opt/llama", "LLAMA_CACHE": CACHE_DIR})
)

app = modal.App("hw10-serve-plain", image=image)
cache = modal.Volume.from_name("hw9-model-cache", create_if_missing=True)


@app.function(
    gpu=GPU_TYPE,
    volumes={CACHE_DIR: cache},
    scaledown_window=SCALEDOWN_SECONDS,
    timeout=900,
    max_containers=1,
    secrets=[modal.Secret.from_name("hw7-api-key")],
)
@modal.concurrent(max_inputs=PARALLEL_SLOTS)
@modal.web_server(PORT, startup_timeout=900)
def serve():
    import os

    api_key = os.environ["LLAMA_API_KEY"]

    subprocess.Popen(
        [
            "/opt/llama/llama-server",
            "-hf",
            f"{MODEL_REPO}:{MODEL_QUANT}",
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--api-key",
            api_key,
            "--ctx-size",
            str(PARALLEL_SLOTS * CONTEXT_PER_SLOT),
            "--parallel",
            str(PARALLEL_SLOTS),
            "--n-gpu-layers",
            "99",
            "--alias",
            "qwen2.5-7b",
        ]
    )
