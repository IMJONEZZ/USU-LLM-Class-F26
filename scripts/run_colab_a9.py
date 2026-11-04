#!/usr/bin/env python3
"""Primary A9 demo path: Colab Pro GPU + Guidance + Qwen.

Always requests a GPU (never CPU). Default is T4 — smallest Colab GPU that
comfortably fits Qwen2.5-1.5B-Instruct in 4-bit (free tier + Pro). If T4 is
busy/unavailable, walks T4 → L4 → A100 → G4 → H100.

Examples:
  make colab
  make colab GPU=A100
  python scripts/run_colab_a9.py --prompt "Short page about Aggie Ice Cream"
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REMOTE_JOB = REPO / "scripts" / "colab_remote_job.py"
LOCAL_OUT_DIR = REPO / "artifacts" / "colab"
SESSION_DEFAULT = "a9-html"

# 3B 4-bit is light on VRAM; T4 (16GB) is the smallest Colab CLI GPU and works
# on free tier. Bigger cards are fallbacks only.
DEFAULT_GPU = "T4"
GPU_FALLBACKS = ("T4", "L4", "A100", "G4", "H100")

PACKAGES = (
    "guidance",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "torch",
)
JOB_TIMEOUT_SEC = "7200"


def colab(
    args: list[str], *, check: bool = True, **kwargs
) -> subprocess.CompletedProcess:
    cmd = ["colab", *args]
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, text=True, **kwargs)


def try_new_gpu_session(session: str, gpu: str) -> bool:
    # Always pass --gpu — we never allocate a CPU runtime for this assignment.
    return (
        subprocess.run(
            ["colab", "new", "-s", session, "--gpu", gpu],
            text=True,
            check=False,
        ).returncode
        == 0
    )


def provision_gpu(session: str, *, preferred: str) -> str:
    """Request preferred GPU, then walk the fallback list. GPU only."""
    order: list[str] = []
    for name in (preferred, *GPU_FALLBACKS):
        if name not in order:
            order.append(name)

    print(f"[local] Colab GPU required. Preferred={preferred!r}, order={order}")
    for candidate in order:
        print(f"[local] requesting --gpu {candidate} …")
        if try_new_gpu_session(session, candidate):
            return candidate
        print(f"[local] {candidate} not available, trying next …")

    raise SystemExit(
        "Could not allocate any Colab GPU (tried "
        + ", ".join(order)
        + "). Check `colab usage` / Pro units, then retry."
    )


def make_src_tarball(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w:gz") as tar:
        tar.add(REPO / "src", arcname="src")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--session", default=SESSION_DEFAULT)
    parser.add_argument(
        "--gpu",
        default=os.environ.get("A9_COLAB_GPU", DEFAULT_GPU),
        help=f"Preferred GPU (default {DEFAULT_GPU}). Always requests a GPU.",
    )
    parser.add_argument(
        "--prompt",
        default="Write a short page about Utah State University for new students.",
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get("A9_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct"),
    )
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--keep", action="store_true", help="Don't stop the VM")
    args = parser.parse_args()

    if shutil.which("colab") is None:
        raise SystemExit("colab CLI missing. Install: uv tool install google-colab-cli")
    if not REMOTE_JOB.is_file():
        raise SystemExit(f"missing {REMOTE_JOB}")

    preferred = (args.gpu or DEFAULT_GPU).strip().upper()
    if preferred not in {"T4", "L4", "G4", "H100", "A100"}:
        raise SystemExit(
            f"Unsupported GPU {preferred!r}. Use one of: T4, L4, G4, H100, A100"
        )

    subprocess.run(["colab", "stop", "-s", args.session], check=False)

    got = provision_gpu(args.session, preferred=preferred)
    print(f"[local] session {args.session!r} ready on GPU {got}")

    tmp = Path(tempfile.mkdtemp(prefix="a9-colab-"))
    try:
        tarball = make_src_tarball(tmp / "a9-src.tar.gz")
        colab(["upload", "-s", args.session, str(tarball), "/content/a9-src.tar.gz"])
        colab(
            ["exec", "-s", args.session, "--timeout", "120"],
            input=(
                "import tarfile\n"
                "from pathlib import Path\n"
                "tarfile.open('/content/a9-src.tar.gz').extractall('/content')\n"
                "print('src ok', sorted(p.name for p in Path('/content/src').iterdir()))\n"
            ),
        )

        colab(["install", "-s", args.session, *PACKAGES])

        load_4bit = "false" if args.no_4bit else "true"
        colab(
            [
                "exec",
                "-s",
                args.session,
                "-f",
                str(REMOTE_JOB),
                "--timeout",
                JOB_TIMEOUT_SEC,
                "--env",
                f"A9_PROMPT={args.prompt}",
                "--env",
                f"A9_MODEL_ID={args.model_id}",
                "--env",
                f"A9_LOAD_IN_4BIT={load_4bit}",
            ]
        )

        LOCAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
        local_html = LOCAL_OUT_DIR / "a9_output.html"
        local_meta = LOCAL_OUT_DIR / "a9_meta.txt"
        colab(
            [
                "download",
                "-s",
                args.session,
                "/content/artifacts/a9_output.html",
                str(local_html),
            ]
        )
        colab(
            [
                "download",
                "-s",
                args.session,
                "/content/artifacts/a9_meta.txt",
                str(local_meta),
            ],
            check=False,
        )

        print(f"[local] saved {local_html}")
        if local_html.is_file():
            text = local_html.read_text(encoding="utf-8")
            print(text[:2000])
            if len(text) > 2000:
                print("…")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if args.keep:
            print(f"[local] keeping session {args.session!r}")
        else:
            print(f"[local] stopping session {args.session!r}")
            subprocess.run(["colab", "stop", "-s", args.session], check=False)


if __name__ == "__main__":
    raise SystemExit(main())
