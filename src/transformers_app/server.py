"""Serve the Assignment 6 final LoRA adapter with FastAPI + Transformers/PEFT.

Start with ``make serve-transformers`` from the repository root.
Model loading happens during startup, so importing this module is inexpensive.

Works on CPU or CUDA. The Unsloth server (src/server.py) reuses create_app()
from here but swaps in a different load_runtime().
"""

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from time import perf_counter

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

# Hugging Face model id + the exact git commit (revision) used when training A6.
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
# Recorded in the final adapter's run.json; keep the base used during training.
MODEL_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"

# Default path to the saved LoRA adapter on disk.
# __file__ = this .py file; parents[2] = repo root (src/transformers_app -> src -> root).
DEFAULT_ADAPTER = (
    Path(__file__).resolve().parents[2]
    / "archive/a6_colab_results/adapters/04_focus2_FINAL"
)
MAX_INPUT_TOKENS = 1024  # reject prompts longer than this after chat formatting


# --- Request / response schemas (Pydantic) ---
# FastAPI uses these to validate JSON bodies and document /docs.


class GenerationRequest(BaseModel):
    """What the client must send in POST /generate."""

    # Strip whitespace; reject unknown JSON fields.
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    prompt: str = Field(min_length=1, max_length=8000)
    max_new_tokens: int = Field(default=128, ge=1, le=256, strict=True)


class GenerationResponse(BaseModel):
    """What /generate returns to the client."""

    text: str  # model answer only (not the prompt)
    prompt_tokens: int
    generated_tokens: int
    elapsed_seconds: float


@dataclass
class Runtime:
    """Holds the loaded model + tokenizer while the server is running.

    One Runtime is created at startup and stored on app.state.runtime.
    """

    model: object
    tokenizer: object
    device: str  # "cpu" or "cuda"
    adapter: Path
    backend: str = "transformers-peft"  # shown by /health (vs "unsloth")
    # Lock = only one generate() at a time (model is big; avoids OOM / races).
    lock: object = field(default_factory=Lock)

    def generate(self, request):
        """Run one greedy generation and package the answer as GenerationResponse."""
        # Serialize generation so concurrent HTTP requests share one model safely.
        # inference_mode() = faster/safer than training mode (no gradient tracking).
        with self.lock, torch.inference_mode():
            started = perf_counter()

            # Wrap the user string in the chat template (roles, special tokens).
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": request.prompt}],
                tokenize=False,
                add_generation_prompt=True,  # leave the model ready to answer
            )
            # The chat template already includes BOS and the role delimiters.
            # return_tensors="pt" => PyTorch tensors; .to(device) moves them to GPU/CPU.
            inputs = self.tokenizer(
                prompt, add_special_tokens=False, return_tensors="pt"
            ).to(self.device)
            prompt_tokens = inputs["input_ids"].shape[-1]
            if prompt_tokens > MAX_INPUT_TOKENS:
                raise ValueError(
                    f"Prompt exceeds {MAX_INPUT_TOKENS} tokens after chat formatting."
                )

            # do_sample=False => greedy decoding (always pick the top token).
            output = self.model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            # output includes the prompt tokens; slice them off so we return only
            # the newly generated answer.
            tokens = output[0, prompt_tokens:]
            return GenerationResponse(
                text=self.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                prompt_tokens=prompt_tokens,
                generated_tokens=len(tokens),
                elapsed_seconds=round(perf_counter() - started, 3),
            )


def load_runtime():
    """Load Transformers base model + PEFT LoRA adapter (CPU or CUDA)."""
    # Keep these imports lazy so the Unsloth server can reuse the API without
    # importing Transformers/PEFT before Unsloth applies its inference patches.
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = (
        Path(os.environ.get("ADAPTER_PATH", DEFAULT_ADAPTER)).expanduser().resolve()
    )
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter / filename).is_file():
            raise FileNotFoundError(f"Missing {adapter / filename}; set ADAPTER_PATH.")

    # Confirm adapter was trained for this base model id.
    config = PeftConfig.from_pretrained(adapter, local_files_only=True)
    if config.base_model_name_or_path != MODEL_ID:
        raise ValueError(f"The adapter must be trained on {MODEL_ID}.")

    # DEVICE env: auto | cpu | cuda
    device = os.environ.get("DEVICE", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError("DEVICE must be auto, cpu, or cuda.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE=cuda requested, but CUDA is unavailable.")

    # Tokenizer from the adapter dir (chat template + special tokens from training).
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Base causal LM (predicts next token). dtype: FP16 on GPU, FP32 on CPU.
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        attn_implementation="sdpa",
    )
    # Loading the base alone would discard all Assignment 6 fine-tuning.
    # PeftModel attaches the LoRA weights on top of the base.
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
    model.to(device)
    model.eval()  # inference mode for the whole model (dropout off, etc.)
    return Runtime(model, tokenizer, device, adapter)


def create_app(loader=None):
    """Build the FastAPI app.

    loader: optional function that returns a Runtime.
      - transformers path: default load_runtime above
      - Unsloth path: src.server.load_runtime passed in
    """

    @asynccontextmanager
    async def lifespan(app):
        # Runs once when Uvicorn starts: load the heavy model into memory.
        app.state.runtime = (loader or load_runtime)()
        try:
            yield  # server is live and handling requests during this stretch
        finally:
            # Runs on shutdown: drop the reference so memory can be freed.
            del app.state.runtime

    app = FastAPI(title="Assignment 7 — LoRA inference", lifespan=lifespan)

    @app.get("/health")
    def health():
        """GET /health — quick status JSON (safe to open in a browser)."""
        runtime = app.state.runtime
        return {
            "status": "ready",
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "adapter": str(runtime.adapter),
            "device": runtime.device,
            "backend": runtime.backend,
        }

    @app.post("/generate", response_model=GenerationResponse)
    def generate(request: GenerationRequest):
        """POST /generate — run the model. Browser GET will get 405; use /docs."""
        # A synchronous handler runs in FastAPI's thread pool; /health stays usable.
        try:
            return app.state.runtime.generate(request)
        except ValueError as exc:
            # e.g. prompt too long after chat template -> HTTP 422
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


# Default app for: uvicorn src.transformers_app.server:app
app = create_app()
