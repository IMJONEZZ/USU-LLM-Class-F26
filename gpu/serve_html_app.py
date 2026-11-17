"""Serves Qwen2.5 7B behind a grammar that only lets it emit HTML.

The grammar in `gpu/html.gbnf` is loaded with `--grammar-file`, which makes it
the server's default sampler instead of something each caller has to remember
to send. A service whose job is returning HTML shouldn't depend on the caller
asking for HTML politely, and it means there's no grammar to keep in sync in
the client.

This is the first GPU server of the semester. Assignments 7 and 8 ran a 1.2B on
CPU at about 20 tokens a second, which was fine for one-sentence answers. A page
is a few hundred tokens and the model is six times bigger, so CPU would have
turned every request into a minute of waiting.

    modal deploy gpu/serve_html_app.py
"""

import subprocess

import modal

CACHE_DIR = "/cache"
PORT = 8080
SCALEDOWN_SECONDS = 60
GPU_TYPE = "T4"

# Bigger model than the last two assignments, because the grammar can force
# well formed markup but it can't make the writing any good. Qwen's own GGUF
# repo splits Q4_K_M across two shards and bartowski's is one file, which is
# one less thing for llama.cpp to piece together.
MODEL_REPO = "bartowski/Qwen2.5-7B-Instruct-GGUF"
MODEL_QUANT = "Q4_K_M"

GRAMMAR_PATH = "/opt/llama/html.gbnf"

LLAMA_CPP_VERSION = "b11636"
LLAMA_CPP_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/"
    f"{LLAMA_CPP_VERSION}/llama-{LLAMA_CPP_VERSION}-bin-ubuntu-cuda-12.8-x64.tar.gz"
)

image = (
    # Not debian_slim this time. The CPU build the last two assignments used
    # runs there fine, but the CUDA build is compiled on Ubuntu 24.04 and wants
    # GLIBC 2.38, and Debian 12 ships 2.36. My first deploy got through the
    # whole image build and then died on startup with "version GLIBC_2.38 not
    # found", which reads like a broken binary and is really a wrong base
    # image. The CUDA runtime image fixes the glibc and brings the CUDA
    # libraries with it, so there's no separate cudart download either.
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
    # Same trap as assignment 7. llama.cpp loads its backends out of sibling
    # .so files and splitting the binary from them gets "no backends are
    # loaded" with nothing else to go on.
    .env({"LD_LIBRARY_PATH": "/opt/llama", "LLAMA_CACHE": CACHE_DIR})
    .add_local_file("gpu/html.gbnf", GRAMMAR_PATH)
)

app = modal.App("hw9-serve-html", image=image)
cache = modal.Volume.from_name("hw9-model-cache", create_if_missing=True)


@app.function(
    gpu=GPU_TYPE,
    volumes={CACHE_DIR: cache},
    scaledown_window=SCALEDOWN_SECONDS,
    timeout=900,
    max_containers=1,
    secrets=[modal.Secret.from_name("hw7-api-key")],
)
@modal.concurrent(max_inputs=4)
# Longer than assignment 8's window. This one may have to pull about 4.7GB off
# Hugging Face the first time, and only the first boot pays for it.
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
            "4096",
            "--n-gpu-layers",
            "99",
            "--grammar-file",
            GRAMMAR_PATH,
            "--alias",
            "qwen2.5-7b-html",
        ]
    )
