"""Run the student's FP16 Llama progression and append evidence to the log."""

import os
import re
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
REPORT = DOCS / "assignment4_local_gpu_results.md"
IMAGE = "usu-llm-a4-local"

MODELS = (
    ("meta-llama/Llama-3.2-1B", "llama_1b"),
    ("meta-llama/Llama-3.2-3B", "llama_3b"),
    ("meta-llama/Meta-Llama-3.1-8B", "llama_8b"),
)


def capture(command):
    """Run a command and return stdout."""
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=True,
    ).stdout


def field(output, label):
    """Extract the last '<label>: <value>' occurrence from model output."""
    matches = re.findall(
        rf"^{re.escape(label)}: (.*)$",
        output,
        re.MULTILINE,
    )
    return matches[-1] if matches else "Not measured / not reached"


def attempt(docker, model_id, slug, run_dir, gpu, image_id):
    """Run one model attempt and append its measured results to the report."""

    command = [
        *docker,
        "run",
        "--rm",
        "--gpus",
        "all",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HF_HOME=/hf-cache",
        "-v",
        f"{Path.home() / '.cache/huggingface'}:/hf-cache",
        "-v",
        f"{ROOT / 'src/gpu_inference.py'}:/app/src/gpu_inference.py:ro",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-e",
        "HF_HUB_DISABLE_PROGRESS_BARS=1",
        "-e",
        "TQDM_DISABLE=1",
        "-e",
        f"MODEL_ID={model_id}",
        "-e",
        "PROMPT=Once upon a time",
        "-e",
        "MAX_NEW_TOKENS=50",
        "-e",
        "TEMPERATURE=0.7",
        "-e",
        "PAUSE_SECONDS=45",
        IMAGE,
    ]

    started = datetime.now(UTC).isoformat()

    raw = run_dir / f"{slug}.txt"
    evidence = run_dir / f"nvidia_smi_{slug}.txt"

    lines = []
    evidence_status = "Not captured: model did not reach the post-load pause."

    with raw.open("w") as log:
        log.write(f"Host start UTC: {started}\nCommand: {shlex.join(command)}\n")
        log.flush()

        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ) as process:
            if process.stdout is None:
                raise RuntimeError("Failed to capture Docker process output.")

            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                lines.append(line)

                if "for nvidia-smi inspection..." in line:
                    try:
                        snapshot = capture(["nvidia-smi"])

                        evidence.write_text(
                            f"Captured UTC: {datetime.now(UTC).isoformat()}\n"
                            + snapshot
                        )

                        evidence_status = (
                            f"[{evidence.name}]({evidence.relative_to(DOCS)})"
                        )

                        print()
                        print(snapshot, flush=True)

                        print(
                            "SCREENSHOT NOW: the model is fully loaded and "
                            "paused for 45 seconds.\n"
                            "Capture this terminal showing nvidia-smi, or run "
                            "nvidia-smi in another terminal and capture it "
                            "before the pause ends.",
                            flush=True,
                        )

                    except subprocess.CalledProcessError as error:
                        evidence_status = f"nvidia-smi failed: {error.stderr.strip()}"

            returncode = process.wait()

        log.write(f"\nDocker exit code: {returncode}\n")

    output = "".join(lines)

    loaded = (
        "=== Model Loaded ===" in output and "Allocated after load bytes:" in output
    )

    success = returncode == 0 and "RESULT: SUCCESS" in output and loaded

    errors = re.findall(
        r"^.*(?:Error|Exception):.*$",
        output,
        re.MULTILINE,
    )

    failure = errors[-1] if errors else f"Docker exit {returncode}; see raw transcript."

    generated = re.search(
        r"=== Generated Text ===\n(.*?)\n=== GPU Memory ===",
        output,
        re.DOTALL,
    )

    rows = {
        "Timestamp (host UTC)": started,
        "Provider/environment": "Local",
        "GPU / physical VRAM / driver (nvidia-smi)": gpu.strip(),
        "Docker image": f"{IMAGE} ({image_id.strip()})",
        "Python version": field(output, "Python"),
        "PyTorch version": field(output, "PyTorch"),
        "Transformers version": field(output, "Transformers"),
        "CUDA build": field(output, "PyTorch CUDA build"),
        "Exact model ID": model_id,
        "Parameter count": field(output, "Parameters"),
        "Precision": "FP16 (requested)",
        "Quantization": "none",
        "CPU offloading": ("no; CPU deserialization precedes full GPU transfer"),
        "Parameter devices": field(output, "Parameter devices"),
        "Buffer devices": field(output, "Buffer devices"),
        "Prompt": "Once upon a time",
        "max_new_tokens": "50",
        "Temperature": "0.7 (sampling; no fixed seed)",
        "Model load result": (
            "Full GPU load succeeded" if loaded else "Failed / not completed"
        ),
        "Generation result": (
            "Completed" if success else "Failed or not reached; see transcript"
        ),
        "Allocated after load (bytes)": field(
            output,
            "Allocated after load bytes",
        ),
        "Reserved after load (bytes)": field(
            output,
            "Reserved after load bytes",
        ),
        "Peak allocated (bytes)": field(
            output,
            "Peak allocated bytes",
        ),
        "Final allocated (bytes)": field(
            output,
            "Final allocated bytes",
        ),
        "Final reserved (bytes)": field(
            output,
            "Final reserved bytes",
        ),
        "Outcome": "SUCCESS" if success else "FAILURE",
        "Docker exit code": str(returncode),
        "Exact failure reason": "None" if success else failure,
        "nvidia-smi evidence": evidence_status,
    }

    with REPORT.open("a") as report:
        report.write(
            f"\n## {started} — {model_id}\n\n| Field | Observed value |\n|---|---|\n"
        )

        for key, value in rows.items():
            value = value.replace("|", "\\|").replace("\n", "<br>")
            report.write(f"| {key} | {value} |\n")

        report.write(
            f"\nRaw transcript and command: [{raw.name}]({raw.relative_to(DOCS)}).\n"
        )

        if generated:
            report.write(
                "\nRepresentative generated output:\n\n"
                "```text\n"
                f"{generated[1].strip()}\n"
                "```\n"
            )
        elif not success:
            report.write("\nNo successful generated output recorded.\n")

    return success


def main():
    """Run all requested Llama checkpoints sequentially."""

    if os.getuid() == 0:
        raise SystemExit(
            "Run as your normal user; the runner uses sudo only for Docker."
        )

    docker = ["docker"]

    probe = subprocess.run(
        [*docker, "image", "inspect", IMAGE],
        capture_output=True,
        text=True,
        check=False,
    )

    if probe.returncode:
        # Let the user's terminal handle sudo authentication.
        # The password is never stored or written to experiment logs.
        subprocess.run(["sudo", "-v"], check=True)
        docker = ["sudo", "-n", "docker"]

    image_id = capture(
        [
            *docker,
            "image",
            "inspect",
            IMAGE,
            "--format",
            "{{.Id}}",
        ]
    )

    gpu = capture(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = DOCS / f"local_gpu_{stamp}"
    run_dir.mkdir()

    (run_dir / "nvidia_smi_before.txt").write_text(capture(["nvidia-smi"]))

    failures = 0

    for index, (model_id, slug) in enumerate(MODELS, start=1):
        print()
        print("=" * 80)
        print(f"STARTING MODEL {index}/{len(MODELS)}: {model_id}")
        print("=" * 80)
        print()

        success = attempt(
            docker,
            model_id,
            slug,
            run_dir,
            gpu,
            image_id,
        )

        print()
        print("=" * 80)

        if success:
            print(f"COMPLETED: {model_id} -> SUCCESS")
            print(
                "The required nvidia-smi screenshot should have been "
                "taken during the 45-second loaded-model pause."
            )
        else:
            print(f"COMPLETED: {model_id} -> FAILURE")
            failures += 1

            input(
                "\nThe model failed. If you want additional evidence, "
                "take a screenshot of the failure now.\n"
                "Press Enter when ready to continue to the next model..."
            )

        print("=" * 80)

    print()
    print("=" * 80)
    print("ALL MODEL ATTEMPTS COMPLETE")
    print("=" * 80)
    print(f"Models attempted: {len(MODELS)}")
    print(f"Successful models: {len(MODELS) - failures}")
    print(f"Failed models: {failures}")
    print(f"Evidence directory: {run_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
