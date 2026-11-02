import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path

import modal

from src import config

VERSIONED_PACKAGES = ["torch", "transformers", "accelerate", "guidance", "pydantic"]
PROBE_PATH = "results/m4_probe.json"
MIB = 2**20

app = modal.App("webpage-benchmark")

hf_volume = modal.Volume.from_name(config.HF_CACHE_VOLUME, create_if_missing=True)

# shortcut: packages are unpinned; the versions are printed at container start, pin them after the first good run
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch", "transformers", "accelerate", "guidance", "pydantic")
    .add_local_dir("src", remote_path="/root/src")
)


def kv_cache_mib(layers, kv_heads, head_dim, tokens, bytes_per_value=2):
    """Memory for the attention cache: one key and one value vector per layer, kv head and token."""
    return 2 * layers * kv_heads * head_dim * bytes_per_value * tokens / MIB


def is_truncated(token_ids, eos_ids, cap):
    """True when generation stopped because it hit the cap, not because the model ended its turn."""
    return len(token_ids) >= cap and token_ids[-1] not in eos_ids


def guidance_answer(model, tokenizer):  # pragma: no cover - needs a GPU machine
    from guidance import gen, models

    lm = models.Transformers(model, tokenizer)
    lm += "The capital of France is " + gen("answer", max_tokens=5, temperature=0.0)
    return lm["answer"]


@app.cls(
    image=image,
    gpu=config.GPU_TYPE,
    volumes={"/root/.cache/huggingface": hf_volume},
    scaledown_window=config.SCALEDOWN_WINDOW,
    timeout=20 * 60,
)
class Generator:
    @modal.enter()
    def load(self):  # pragma: no cover - needs a GPU machine
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        versions = {
            name: importlib.metadata.version(name) for name in VERSIONED_PACKAGES
        }
        print("versions:", versions)
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        # plain transformers, not Unsloth: Guidance cannot read Unsloth's patched cache (M4 diagnosis)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME, dtype=torch.bfloat16, device_map={"": 0}
        )
        hf_volume.commit()  # keep the downloaded weights for the next cold start
        self.versions = versions
        self.container_id = os.environ.get("MODAL_TASK_ID", "unknown")
        self.started = time.time()
        self.calls = {}

    def meta(self, method):  # pragma: no cover - needs a GPU machine
        before = self.calls.get(method, 0)
        self.calls[method] = before + 1
        return {
            "container_id": self.container_id,
            "container_age_s": round(time.time() - self.started, 2),
            "calls_before": before,
        }

    def encode(self, text):  # pragma: no cover - needs a GPU machine
        chat = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=False,
        )
        # the chat template already holds the start token
        return self.tokenizer(chat, return_tensors="pt", add_special_tokens=False).to(
            "cuda"
        )

    @modal.method()
    def probe(self):  # pragma: no cover - needs a GPU machine
        import torch

        from src.schema import WebPage

        # assumed shape of Method B's prompt (M6 writes the real builder): base template + schema text
        schema_text = json.dumps(WebPage.model_json_schema())
        prompts = [
            config.BASE_PROMPT_TEMPLATE.format(request=p) + "\n" + schema_text
            for p in config.PROMPTS
        ]
        prompt_tokens = max(self.encode(p)["input_ids"].shape[1] for p in prompts)
        total_tokens = prompt_tokens + config.MAX_NEW_TOKENS
        cfg = self.model.config
        head_dim = (
            getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        )
        params = sum(p.numel() for p in self.model.parameters())
        element_bytes = next(self.model.parameters()).element_size()
        gpu_name = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        estimate_weights = params * element_bytes / MIB
        estimate_kv = kv_cache_mib(
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            head_dim,
            total_tokens,
            element_bytes,
        )
        return {
            "model": config.MODEL_NAME,
            "gpu": gpu_name,
            "gpu_total_mib": round(
                torch.cuda.get_device_properties(0).total_memory / MIB, 2
            ),
            "versions": self.versions,
            "params": params,
            "element_bytes": element_bytes,
            "max_prompt_tokens": prompt_tokens,
            "max_total_tokens": total_tokens,
            "estimate_weights_mib": round(estimate_weights, 2),
            "estimate_kv_cache_mib": round(estimate_kv, 2),
            "measured_allocated_mib": round(torch.cuda.memory_allocated() / MIB, 2),
            "measured_peak_mib": round(torch.cuda.max_memory_allocated() / MIB, 2),
        }

    @modal.method()
    def generate(self, prompt):  # pragma: no cover - needs a GPU machine
        import torch

        start = time.perf_counter()
        inputs = self.encode(prompt)
        sampling = {"temperature": config.TEMPERATURE} if config.TEMPERATURE > 0 else {}
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                do_sample=config.TEMPERATURE > 0,
                **sampling,
            )
        new_ids = output[0][inputs["input_ids"].shape[1] :].tolist()
        seconds = time.perf_counter() - start
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        return {
            "text": self.tokenizer.decode(new_ids, skip_special_tokens=True),
            "new_tokens": len(new_ids),
            "hit_cap": is_truncated(new_ids, eos_ids, config.MAX_NEW_TOKENS),
            "seconds": round(seconds, 2),
            **self.meta("generate"),
        }

    @modal.method()
    def guidance_check(self):  # pragma: no cover - needs a GPU machine
        """Can Guidance drive this model? Failures are left to surface."""
        return guidance_answer(self.model, self.tokenizer)


@app.local_entrypoint()
def smoke():  # pragma: no cover - needs Modal and a GPU
    # smoke only: fall back to an A10 if no L4 is free; the benchmark uses the class default (L4)
    generator = Generator.with_options(gpu=["L4", "A10"])()
    info = generator.probe.remote()
    print(json.dumps(info, indent=2))
    samples = [generator.generate.remote(p) for p in config.PROMPTS[:2]]
    for sample in samples:
        print(
            sample["new_tokens"],
            "tokens",
            sample["seconds"],
            "s hit_cap:",
            sample["hit_cap"],
        )
        print(sample["text"][:400])
    answer = generator.guidance_check.remote()
    print("guidance answer:", answer)
    Path(PROBE_PATH).parent.mkdir(exist_ok=True)
    Path(PROBE_PATH).write_text(
        json.dumps(
            {"probe": info, "samples": samples, "guidance_answer": answer}, indent=2
        )
    )
