import subprocess
import threading
import warnings

import modal
from modal import FilePatternMatcher

app = modal.App("my_apps_name")
GPU_TYPE = "T4"


def build_image() -> modal.Image:
    return modal.Image.debian_slim(python_version="3.11").uv_pip_install("unsloth")


def build_test_image() -> modal.Image:
    return (
        build_image()
        .uv_pip_install("pytest", "pytest-cov")
        .add_local_dir(
            ".", remote_path="/root", ignore=FilePatternMatcher.from_file(".gitignore")
        )
    )


def log_gpu_usage(stop_event: threading.Event, interval: float = 5.0) -> None:
    while not stop_event.is_set():
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        used_str, total_str = result.stdout.strip().split(",")
        used_mib = float(used_str.strip())
        total_mib = float(total_str.strip())
        percent = (used_mib / total_mib) * 100
        print(f"[gpu usage] {used_mib:.0f}/{total_mib:.0f} MiB ({percent:.1f}%)")
        stop_event.wait(interval)


def load_model(model_name: str, bit_width: str = "16bit"):
    if bit_width == "16bit":
        load_in_16bit, load_in_4bit = True, False
    elif bit_width == "4bit":
        load_in_16bit, load_in_4bit = False, True
    else:
        raise ValueError(
            f"Unsupported bit_width: {bit_width!r}. Use '16bit' or '4bit'."
        )

    warnings.filterwarnings("ignore", message=".*inline_inbuilt_nn_modules.*")

    from unsloth import FastLanguageModel

    stop_event = threading.Event()
    monitor = threading.Thread(target=log_gpu_usage, args=(stop_event,))
    monitor.start()
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            load_in_16bit=load_in_16bit,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
        model.generation_config.max_length = None
    finally:
        stop_event.set()
        monitor.join()

    return model, tokenizer


@app.function(
    gpu=GPU_TYPE,
    image=build_image(),
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate(model_name: str, bit_width: str = "16bit") -> bool:
    import torch

    try:
        model, tokenizer = load_model(model_name, bit_width)

        from transformers import set_seed

        set_seed(42)
        inputs = tokenizer("Once upon a time", return_tensors="pt").to("cuda")
        output = model.generate(
            **inputs, temperature=1, do_sample=True, max_new_tokens=50
        )
        print(tokenizer.decode(output[0], skip_special_tokens=True))
        return True
    except torch.cuda.OutOfMemoryError:
        print(
            f"[fit check] FAILED: {model_name} ({bit_width}) does not fit on {GPU_TYPE} (out of memory)"
        )
        return False


@app.local_entrypoint()
def main() -> None:
    generate.remote(model_name="unsloth/Llama-3.2-1B", bit_width="4bit")


@app.function(gpu=GPU_TYPE, image=build_test_image())
def run_tests() -> None:
    import subprocess

    result = subprocess.run(
        ["pytest", "-vs"], cwd="/root", capture_output=True, text=True, check=False
    )
    print(result.stdout)
    print(result.stderr)
    result.check_returncode()
