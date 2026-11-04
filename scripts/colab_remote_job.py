"""Runs on the Colab VM after ``src/`` has been uploaded to ``/content/src``.

Loads Qwen through Guidance, generates constrained HTML, writes artifacts under
``/content/artifacts/``.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

CONTENT = Path("/content")
SRC_ROOT = CONTENT  # so ``import src...`` works after we extract the tarball here
ARTIFACTS = CONTENT / "artifacts"
OUT_HTML = ARTIFACTS / "a9_output.html"
OUT_META = ARTIFACTS / "a9_meta.txt"

DEFAULT_PROMPT = "Write a short page about Utah State University for new students."


def main() -> int:
    prompt = os.environ.get("A9_PROMPT", DEFAULT_PROMPT).strip() or DEFAULT_PROMPT
    model_id = os.environ.get("A9_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    # 4-bit is the sane default on Colab T4/L4 unless you explicitly turn it off.
    os.environ.setdefault("A9_LOAD_IN_4BIT", "true")
    os.environ["A9_MODEL_ID"] = model_id

    sys.path.insert(0, str(SRC_ROOT))

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    import torch

    print("[a9] torch", torch.__version__)
    print("[a9] cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[a9] device:", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print(f"[a9] vram_gb: {props.total_memory / 1e9:.1f}")
    else:
        print("[a9] WARNING: no CUDA — Guidance+Transformers will be slow/OOM-prone")

    from src.html_generator import HtmlGenerator, load_guidance_model

    print(f"[a9] loading model {model_id!r} …")
    base = load_guidance_model(model_id)
    generator = HtmlGenerator(model_factory=lambda: base, temperature=0.7)

    print(f"[a9] prompt: {prompt!r}")
    html = generator.generate(prompt)
    OUT_HTML.write_text(html, encoding="utf-8")

    meta = "\n".join(
        [
            f"model_id={model_id}",
            f"load_in_4bit={os.environ.get('A9_LOAD_IN_4BIT')}",
            f"cuda={torch.cuda.is_available()}",
            f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
            f"prompt={prompt}",
            f"html_bytes={len(html.encode('utf-8'))}",
            "",
        ]
    )
    OUT_META.write_text(meta, encoding="utf-8")

    print("[a9] wrote", OUT_HTML)
    print("[a9] ---- HTML preview (first 2k) ----")
    print(html[:2000])
    print("[a9] ---- end preview ----")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:  # noqa: BLE001 - show full remote traceback before exiting
        traceback.print_exc()
        raise SystemExit(1)
