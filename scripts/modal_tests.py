import modal
from modal import FilePatternMatcher

app = modal.App("resume-gpu-tests")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.12.1",
        "torchao==0.18.0",
        "unsloth==2026.9.14",
        "transformers==5.5.0",
        "sentence-transformers==6.1.0",
        "datasets==4.3.0",
        "pandas==3.0.6",
        "numpy==2.4.6",
        "tqdm",
        "pytest",
        "pytest-cov",
        "truststore",
    )
    .env({"HF_HOME": "/hf_cache"})
    .add_local_dir(
        ".", remote_path="/root", ignore=FilePatternMatcher.from_file(".gitignore")
    )
)


@app.function(gpu="T4", image=image, volumes={"/hf_cache": hf_cache}, timeout=1800)
def run_gpu_tests():
    import subprocess

    result = subprocess.run(
        ["pytest", "-vs", "--no-cov"],
        cwd="/root",
        capture_output=True,
        text=True,
        check=False,
    )
    print(result.stdout)
    print(result.stderr)
    hf_cache.commit()
    result.check_returncode()


@app.local_entrypoint()
def main():
    run_gpu_tests.remote()
