"""Serves the stock Llama 3.2 1B for the extra credit.

The extra credit says to use the base model, not the one I fine tuned in
assignment 6, so this is deliberately a separate deployment from
`gpu/serve_app.py`. That one mounts my adapter off a volume. This one pulls an
untouched Llama 3.2 1B Instruct straight from Hugging Face and never sees my
weights at all, which is the only way I can honestly claim the retrieval is
doing the work.

llama-server takes `-hf` and downloads the GGUF itself, so there's no export
step and nothing of mine in the path. The model sits in a cache volume so the
download happens once instead of on every cold start.

    modal deploy gpu/serve_base_app.py
"""

import subprocess

import modal

CACHE_DIR = "/cache"
PORT = 8080
SCALEDOWN_SECONDS = 60

# Stock weights, not mine. Unsloth's repo is a straight GGUF conversion of
# meta-llama/Llama-3.2-1B-Instruct and isn't gated.
BASE_REPO = "unsloth/Llama-3.2-1B-Instruct-GGUF"
BASE_QUANT = "Q4_K_M"

LLAMA_CPP_VERSION = "b11636"
LLAMA_CPP_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/"
    f"{LLAMA_CPP_VERSION}/llama-{LLAMA_CPP_VERSION}-bin-ubuntu-x64.tar.gz"
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "libgomp1", "libcurl4-openssl-dev")
    .run_commands(
        "mkdir -p /opt/llama",
        f"curl -fsSL {LLAMA_CPP_URL} -o /tmp/llama.tar.gz",
        "tar -xzf /tmp/llama.tar.gz -C /opt/llama --strip-components=1",
        "rm /tmp/llama.tar.gz",
        "/opt/llama/llama-server --version",
    )
    # Same trap as last assignment: llama.cpp loads its CPU backend out of a
    # sibling .so and splitting the binary from those gets "no backends are
    # loaded" with nothing else to go on.
    .env({"LD_LIBRARY_PATH": "/opt/llama", "LLAMA_CACHE": CACHE_DIR})
)

app = modal.App("hw8-serve-base", image=image)
cache = modal.Volume.from_name("hw8-base-cache", create_if_missing=True)


@app.function(
    volumes={CACHE_DIR: cache},
    scaledown_window=SCALEDOWN_SECONDS,
    timeout=900,
    max_containers=1,
    secrets=[modal.Secret.from_name("hw7-api-key")],
)
@modal.concurrent(max_inputs=4)
# Twice the startup window last assignment used. That one mounted a GGUF that
# was already sitting on a volume; this one may have to pull 800MB off Hugging
# Face before llama.cpp can open it, and only the first boot pays that.
@modal.web_server(PORT, startup_timeout=600)
def serve():
    import os

    # Reusing last assignment's key instead of minting a second one. The
    # endpoint is public either way and one shared key is already the weakest
    # part of this; two of them wouldn't be an improvement.
    api_key = os.environ["LLAMA_API_KEY"]

    subprocess.Popen(
        [
            "/opt/llama/llama-server",
            "-hf",
            f"{BASE_REPO}:{BASE_QUANT}",
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--api-key",
            api_key,
            "--ctx-size",
            "4096",
            "--threads",
            "-1",
            "--alias",
            "llama-3.2-1b-base",
        ]
    )
