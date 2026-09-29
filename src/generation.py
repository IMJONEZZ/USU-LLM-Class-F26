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


@app.function(gpu="T4", image=build_test_image())
def run_tests() -> None:
    import subprocess

    result = subprocess.run(
        ["pytest", "-vs"], cwd="/root", capture_output=True, text=True, check=False
    )
    result.check_returncode()
