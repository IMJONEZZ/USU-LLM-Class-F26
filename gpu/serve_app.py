"""Serves the assignment 6 model with llama.cpp behind a Modal URL.

`gpu/export_app.py` writes a quantized GGUF onto the `hw7-models` volume. This
mounts that volume, starts llama.cpp's `llama-server` against it, and lets
Modal expose the port over HTTPS. The result speaks the OpenAI chat completions
API, so `src/inference.py` is an ordinary `requests` call.

This runs on CPU. Modal's own serving guides reach for H100s because they're
sized for models a hundred times bigger than mine; a 1.2B at four bits is about
800MB and doesn't need a GPU to answer one caller at a time. If tokens per
second turn out to be embarrassing the fix is one line, GPU_TYPE below.

Scaledown is 60 seconds rather than the 900 those guides use, so an idle
deployment stops costing anything about a minute after the last request.

    modal deploy gpu/serve_app.py
"""

import subprocess

import modal

MODEL_DIR = "/models"
PORT = 8080
# How long a container sticks around after the last request. Modal's vLLM guide
# leaves this at 900 seconds, which on a GPU is fifteen minutes of paying for
# nothing.
SCALEDOWN_SECONDS = 60

# Prebuilt llama.cpp. Building from source works but adds several minutes to
# every image rebuild for a binary that ships ready to run. Pinned so a new
# upstream release can't change what I deployed.
LLAMA_CPP_VERSION = "b11636"
LLAMA_CPP_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/"
    f"{LLAMA_CPP_VERSION}/llama-{LLAMA_CPP_VERSION}-bin-ubuntu-x64.tar.gz"
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "libgomp1")
    .run_commands(
        "mkdir -p /opt/llama",
        f"curl -fsSL {LLAMA_CPP_URL} -o /tmp/llama.tar.gz",
        "tar -xzf /tmp/llama.tar.gz -C /opt/llama --strip-components=1",
        "rm /tmp/llama.tar.gz",
        "/opt/llama/llama-server --version",
    )
    # llama.cpp picks a CPU backend at runtime out of a pile of
    # libggml-cpu-<microarch>.so files, and it looks for them next to the
    # binary. My first attempt copied llama-server into /usr/local/bin and the
    # shared objects into /usr/local/lib, which split it from its backends and
    # got "no backends are loaded" with no other explanation. Leaving the
    # archive as one directory and pointing the loader at it is the fix.
    .env({"LD_LIBRARY_PATH": "/opt/llama"})
)

app = modal.App("hw7-serve", image=image)
models = modal.Volume.from_name("hw7-models", create_if_missing=True)


def find_model(directory=f"{MODEL_DIR}/gguf", quantization="Q4_K_M"):
    """Pick the quantized GGUF out of the export directory.

    The export leaves an F16 intermediate next to the quantized file. Taking
    whichever one the filesystem lists first picked the 2.5GB F16 the first
    time I ran it, so the choice is spelled out here instead of assumed.
    """
    import os

    candidates = sorted(f for f in os.listdir(directory) if f.endswith(".gguf"))
    if not candidates:
        raise FileNotFoundError(f"no .gguf in {directory}")
    preferred = [f for f in candidates if quantization.upper() in f.upper()]
    return os.path.join(directory, (preferred or candidates)[0])


@app.function(
    volumes={MODEL_DIR: models},
    scaledown_window=SCALEDOWN_SECONDS,
    timeout=600,
    max_containers=1,
)
@modal.concurrent(max_inputs=4)
@modal.web_server(PORT, startup_timeout=300)
def serve():
    model_path = find_model()
    print(f"serving {model_path}")

    # --host 0.0.0.0 matters. Bound to localhost the process starts fine and
    # Modal can't route anything to it, which looks like a dead endpoint.
    subprocess.Popen(
        [
            "/opt/llama/llama-server",
            "--model",
            model_path,
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--ctx-size",
            "2048",
            "--threads",
            "-1",
            "--alias",
            "hw6-trick-questions",
        ]
    )
