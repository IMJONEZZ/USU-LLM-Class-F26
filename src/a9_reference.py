"""Optional A9 GPU reference: preserved Guidance/Transformers backend and experiment runner."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import requests
import torch
from guidance import assistant, system, user
from guidance.models import Transformers
from transformers import BitsAndBytesConfig

from src.html_evaluation import (
    HtmlValidationError,
    evaluate_html,
    make_html,
    validate_html,
)

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Reminds the model that *we* own the tags — it should just write content.
SYSTEM_PROMPT = (
    "You are an expert in HTML. Produce concise, factual page content. "
    "Do not invent scripts, event handlers, or attributes; the host grammar "
    "inserts tags for you."
)


class HtmlGenerationError(RuntimeError):
    """Generation or post-generation validation failed."""


@dataclass
class HtmlGenerator:
    """Thin wrapper: prompt in, validated HTML out.

    ``model_factory`` returns a Guidance model (real Transformers, Mock in tests).
    """

    model_factory: Callable[[], Any]
    temperature: float = 0.7

    def generate(self, prompt: str) -> str:
        cleaned = prompt.strip()
        if not cleaned:
            raise HtmlGenerationError("Prompt must not be empty")

        try:
            document = self._generate_with_guidance(cleaned)
        except HtmlGenerationError:
            raise
        except Exception as exc:
            # Surface Guidance/torch surprises as our error type.
            raise HtmlGenerationError(str(exc)) from exc

        # Tags/nesting were forced during decode. Whitelist validation is the
        # hard response gate. We do not require html_grammar().match() here:
        # with a regex, Guidance's gen(max_tokens=...) can soft-exceed and then
        # fail a strict rematch even though decode was constrained.
        try:
            return validate_html(document)
        except HtmlValidationError as exc:
            raise HtmlGenerationError(str(exc)) from exc

    def _generate_with_guidance(self, prompt: str) -> str:
        lm = self.model_factory()
        # Shared Transformers engine keeps a KV cache across requests; clear it
        # so a prior run can't blow up with mismatched attention sizes.
        _reset_guidance_engine_cache(lm)
        # Guidance models are immutable — each += returns a new state object.
        with system():
            lm += SYSTEM_PROMPT
        with user():
            lm += prompt
        with assistant():
            lm += make_html(name="html_text", temperature=self.temperature)

        document = lm["html_text"]
        if not isinstance(document, str) or not document.strip():
            raise HtmlGenerationError("Guidance returned an empty HTML capture")
        return document


def _reset_guidance_engine_cache(lm: Any) -> None:
    """Drop stale past_key_values / logits on the shared Guidance engine."""
    engine = getattr(lm, "engine", None)
    if engine is None:
        return
    engine._past_key_values = None
    engine._cached_logits = None
    if hasattr(engine, "_cached_token_ids"):
        engine._cached_token_ids = []


def load_guidance_model(
    model_id: str | None = None, *, gpu_only: bool = False
) -> Transformers:
    """Load Qwen through Guidance's Transformers backend (slow the first time)."""
    resolved = model_id or os.environ.get("A9_MODEL_ID", DEFAULT_MODEL_ID)
    kwargs: dict[str, Any] = {
        "echo": False,  # quieter logs in the server
        # Guidance's logits path expects float16/float32 — default bf16 on many
        # GPUs blows up with: mat1/mat2 Float vs BFloat16.
        "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        # Newer transformers Cache.crop() disagrees with Guidance backtracking /
        # fast-forward ("tensor a must match tensor b" on dim 3). Stay correct
        # over the speed tricks.
        "enable_backtrack": False,
        "enable_ff_tokens": False,
    }
    if gpu_only:
        # Bind all weights explicitly so this reference cannot silently offload
        # layers to CPU when GPU memory is insufficient.
        if not torch.cuda.is_available():
            raise RuntimeError("A9 reference requires CUDA; CPU fallback is forbidden")
        kwargs["device_map"] = {"": "cuda:0"}
    load_in_4bit = os.environ.get("A9_LOAD_IN_4BIT", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if load_in_4bit and torch.cuda.is_available():
        # Current Transformers accepts a quantization config instead of the old
        # load_in_4bit keyword; the Guidance grammar and sampling path are unchanged.
        # Handy on Colab when VRAM is tight; needs bitsandbytes.
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
    return Transformers(resolved, **kwargs)


def run_reference(a10=Path("artifacts/a10"), output=Path("artifacts/a9_reference")):
    config = json.loads((a10 / "evaluation_config.json").read_text())
    examples = json.loads((a10 / "held_out.json").read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("A9 reference requires CUDA; CPU fallback is forbidden")
    if (output / "responses.jsonl").exists():
        raise FileExistsError(
            "Reference results already exist; archive before rerunning"
        )
    # Loading two models on the 4 GB T2000 is unnecessary. This unloads, not generates.
    response = requests.post(
        config["api_base"] + "/api/generate",
        json={
            "model": config["model"],
            "keep_alive": 0,
            "stream": False,
        },
        timeout=30,
    )
    response.raise_for_status()
    os.environ["A9_LOAD_IN_4BIT"] = "true"
    base = load_guidance_model(DEFAULT_MODEL_ID, gpu_only=True)
    model = base.engine.model_obj
    devices = {p.device.type for p in model.parameters()}
    # CUDA availability alone does not prove that all weights were placed on GPU.
    if devices != {"cuda"}:
        raise RuntimeError(f"All model parameters must be on CUDA, found {devices}")
    generator = HtmlGenerator(
        model_factory=lambda: base, temperature=config["temperature"]
    )
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": DEFAULT_MODEL_ID,
        "model_commit": model.config._commit_hash,
        "backend": "Guidance/Transformers",
        "gpu": torch.cuda.get_device_name(0),
        "device": str(model.device),
        "parameter_devices": sorted(devices),
        "load_in_4bit": True,
        "temperature": config["temperature"],
        "seeds": config["seeds"],
        "packages": {
            name: importlib.metadata.version(name)
            for name in [
                "torch",
                "transformers",
                "guidance",
                "bitsandbytes",
                "accelerate",
            ]
        },
        "held_out_sha256": hashlib.sha256(
            (a10 / "held_out.json").read_bytes()
        ).hexdigest(),
        "evaluator_sha256": hashlib.sha256(
            Path("src/html_evaluation.py").read_bytes()
        ).hexdigest(),
        "generation_policy": "Original A9 grammar and local limits; no Ollama global token budget. Raw capture scored before response validation gate, without repairs.",
        "config": config,
    }
    (output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    started = time.monotonic()
    records = []
    for example in examples:
        for seed in config["seeds"]:
            # Guidance samples with NumPy; seed it as well as Torch and Python.
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            sample_started = time.monotonic()
            error = None
            try:
                # This is the original generation method. The shared evaluator invokes
                # validate_html afterwards, so rejected captures are also preserved.
                document = generator._generate_with_guidance(example["request"])
            except torch.cuda.OutOfMemoryError:
                raise  # Infrastructure failure: do not replace it with CPU inference.
            except Exception as exc:  # noqa: BLE001 -- record model failures without discarding samples.
                document = ""
                error = f"{type(exc).__name__}: {exc}"
            scores = evaluate_html(document, example["requirements"])
            if error:
                scores = {
                    **scores,
                    "score": 0.0,
                    "instruction_score": 0.0,
                    "instruction_pass": False,
                }
            record = {
                "program": "a9_reference",
                "request_id": example["id"],
                "request": example["request"],
                "seed": seed,
                "html": document,
                "error": error,
                "elapsed_seconds": time.monotonic() - sample_started,
                **scores,
            }
            records.append(record)
            (output / f"{example['id']}-{seed}.html").write_text(document)
            with (output / "responses.jsonl").open("a") as stream:
                # Keep failed attempts in the denominator rather than reporting
                # only the constrained generator's successful responses.
                stream.write(json.dumps(record) + "\n")
            print(
                example["id"],
                seed,
                round(scores["score"], 3),
                error or "generated",
                flush=True,
            )
    metadata["elapsed_seconds"] = time.monotonic() - started
    metadata["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
    (output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summary = {
        key: mean(row[key] for row in records)
        for key in ("score", "grammar_pass", "instruction_pass", "tag_nesting_valid")
    }
    summary.update(samples=len(records), errors=sum(bool(r["error"]) for r in records))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    run_reference()
