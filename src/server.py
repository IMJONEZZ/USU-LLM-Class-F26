"""Serve the final Assignment 6 LoRA adapter with Unsloth: ``make serve``.

This file only loads the model with Unsloth. The HTTP routes (/health, /generate)
live in transformers_app — we reuse them via create_app(load_runtime).
Needs an NVIDIA GPU (CUDA). For CPU, use: make serve-transformers
"""

import json
import os
from pathlib import Path

import torch

# Shared API pieces from the Transformers server package (same endpoints, same
# request/response shapes). We swap in our own load_runtime below.
from src.transformers_app.server import (
    DEFAULT_ADAPTER,
    MAX_INPUT_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    Runtime,
    create_app,
)


def load_runtime():
    """Load base Llama + LoRA adapter using Unsloth, return a Runtime for FastAPI."""

    # ADAPTER_PATH env var overrides the default Assignment 6 adapter folder.
    # expanduser() turns ~ into your home dir; resolve() makes an absolute path.
    adapter = (
        Path(os.environ.get("ADAPTER_PATH", DEFAULT_ADAPTER)).expanduser().resolve()
    )

    # Refuse to start if the adapter directory is incomplete (avoids serving base-only).
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter / filename).is_file():
            raise FileNotFoundError(f"Missing {adapter / filename}; set ADAPTER_PATH.")

    # Read adapter config JSON and check it was trained on the expected base model.
    config = json.loads((adapter / "adapter_config.json").read_text())
    if config.get("base_model_name_or_path") != MODEL_ID:
        raise ValueError(f"The adapter must be trained on {MODEL_ID}.")

    # Unsloth is GPU-only in this project.
    if os.environ.get("DEVICE", "auto") not in ("auto", "cuda"):
        raise ValueError(
            "Unsloth requires DEVICE=auto or cuda; use make serve-transformers for CPU."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Unsloth requires CUDA; use make serve-transformers for CPU."
        )

    # IMPORTANT: import Unsloth BEFORE transformers/peft so it can patch them.
    # The blank "# isort: split" line below stops the import sorter from reordering.
    from unsloth import FastLanguageModel

    # isort: split
    from peft import PeftModel
    from transformers import AutoTokenizer

    # 1) Load the base model at the exact Hugging Face revision used in training.
    #    (Loading only the adapter folder would not pin that revision cleanly.)
    base, _ = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        revision=MODEL_REVISION,
        max_seq_length=MAX_INPUT_TOKENS + 256,
        dtype=torch.float16,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        device_map={"": 0},  # put the whole model on GPU 0
        use_exact_model_name=True,
        use_gradient_checkpointing=False,
    )
    # 2) Attach the LoRA adapter (PEFT) — this is the Assignment 6 fine-tuning.
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
    # 3) Unsloth patches + inference mode helpers.
    model = FastLanguageModel.patch_peft_model(model, use_gradient_checkpointing=False)
    FastLanguageModel.for_inference(model)
    model.requires_grad_(False)  # we are not training; freeze gradients

    # Tokenizer comes from the saved adapter folder (includes chat template).
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
    if tokenizer.pad_token_id is None:
        # Some chat models have no pad token; reuse EOS so batching/padding works.
        tokenizer.pad_token = tokenizer.eos_token

    # Runtime is the object FastAPI stores and calls for /generate and /health.
    return Runtime(model, tokenizer, "cuda", adapter, backend="unsloth")


# Uvicorn looks for `app` in this module: src.server:app
# create_app(load_runtime) wires our Unsloth loader into the shared FastAPI routes.
app = create_app(load_runtime)
