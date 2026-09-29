import modal
from modal import FilePatternMatcher

app = modal.App("my_apps_name")


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


def load_model(model_name: str = "unsloth/Llama-3.2-1B"):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name, load_in_16bit=True, load_in_4bit=False
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


@app.function(gpu="T4", image=build_image())
def generate() -> None:
    from transformers import set_seed

    model, tokenizer = load_model()

    set_seed(42)
    inputs = tokenizer("Once upon a time", return_tensors="pt").to("cuda")
    output = model.generate(**inputs, temperature=1, do_sample=True)
    print(tokenizer.decode(output[0], skip_special_tokens=True))


@app.local_entrypoint()
def main() -> None:
    generate.remote()


@app.function(gpu="T4", image=build_test_image())
def run_tests() -> None:
    import subprocess

    result = subprocess.run(
        ["pytest", "-vs"], cwd="/root", capture_output=True, text=True, check=False
    )
    result.check_returncode()
