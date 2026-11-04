"""Run Guidance + Qwen to produce grammar-constrained HTML."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from guidance import assistant, system, user
from guidance.models import Transformers

from src.html_grammar import make_html
from src.html_validate import HtmlValidationError, validate_html

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Reminds the model that *we* own the tags — it should just write content.
SYSTEM_PROMPT = (
    "You are an expert in HTML. Produce concise, factual page content. "
    "Do not invent scripts, event handlers, or attributes; the host grammar "
    "inserts tags for you."
)


class HtmlGenerationError(RuntimeError):
    """Anything that should become HTTP 422 from the API layer."""


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


def load_guidance_model(model_id: str | None = None) -> Transformers:
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
    load_in_4bit = os.environ.get("A9_LOAD_IN_4BIT", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if load_in_4bit and torch.cuda.is_available():
        # Handy on Colab when VRAM is tight; needs bitsandbytes.
        kwargs["load_in_4bit"] = True
    return Transformers(resolved, **kwargs)


def build_default_generator() -> HtmlGenerator:
    """FastAPI lifespan helper — load weights once, reuse the engine."""

    holder: dict[str, Transformers] = {}

    def factory() -> Transformers:
        if "base" not in holder:
            holder["base"] = load_guidance_model()
        # Same engine, fresh conversation state per request via += above.
        return holder["base"]

    return HtmlGenerator(model_factory=factory)
