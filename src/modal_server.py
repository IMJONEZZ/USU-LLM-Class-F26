import importlib.metadata
import os
import subprocess
import time

import modal

from src import benchmark, config, methods

VERSIONED_PACKAGES = ["torch", "transformers", "accelerate", "guidance", "pydantic"]
RESULTS_DIR = "/results"  # where the results volume is mounted in the container
MIB = 2**20

app = modal.App("webpage-benchmark")

hf_volume = modal.Volume.from_name(config.HF_CACHE_VOLUME, create_if_missing=True)
results_volume = modal.Volume.from_name(config.RESULTS_VOLUME, create_if_missing=True)

# pinned to the versions the last successful smoke run printed (results/m4_probe.json)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.15.0",
        "transformers==5.22.1",
        "accelerate==1.15.0",
        "guidance==0.3.1",
        "pydantic==2.14.2",
    )
    .add_local_dir("src", remote_path="/root/src")
)


def kv_cache_mib(layers, kv_heads, head_dim, tokens, bytes_per_value=2):
    """Memory for the attention cache: one key and one value vector per layer, kv head and token."""
    return 2 * layers * kv_heads * head_dim * bytes_per_value * tokens / MIB


@app.cls(
    image=image,
    gpu=config.GPU_TYPE,
    volumes={"/root/.cache/huggingface": hf_volume, RESULTS_DIR: results_volume},
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
        self.gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        self.container_id = os.environ.get("MODAL_TASK_ID", "unknown")
        self.started = time.time()
        self.calls = {}

    def meta(self, method):
        before = self.calls.get(method, 0)
        self.calls[method] = before + 1
        return {
            "container_id": self.container_id,
            "container_age_s": round(time.time() - self.started, 2),
            "calls_before": before,
        }

    def chat_text(self, text):
        """The one chat-formatted prompt string both methods start from."""
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=False,
        )

    def encode(self, text):  # pragma: no cover - needs a GPU machine
        # the chat template already holds the start token
        return self.tokenizer(
            self.chat_text(text), return_tensors="pt", add_special_tokens=False
        ).to("cuda")

    def probe(self):  # pragma: no cover - needs a GPU machine
        import torch

        prompts = [methods.build_prompt(p) for p in config.PROMPTS]
        prompt_tokens = max(
            len(
                self.tokenizer(self.chat_text(p), add_special_tokens=False)["input_ids"]
            )
            for p in prompts
        )
        total_tokens = prompt_tokens + config.MAX_NEW_TOKENS
        cfg = self.model.config
        head_dim = (
            getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        )
        params = sum(p.numel() for p in self.model.parameters())
        element_bytes = next(self.model.parameters()).element_size()
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
            "gpu": self.gpu,
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

    def complete(self, prompt):  # pragma: no cover - needs a GPU machine
        """Method B's model call: one plain transformers generation."""
        import torch

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
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return (
            text,
            len(new_ids),
            methods.is_truncated(new_ids, eos_ids, config.MAX_NEW_TOKENS),
        )

    def record(self, method, outcome, html, result, seconds):
        bucket, reason = outcome
        return {
            "method": method,
            "bucket": bucket,
            "reason": reason,
            "html": html,
            **result,
            "seconds": round(seconds, 2),
            "model": config.MODEL_NAME,
            "gpu": self.gpu,
            **self.meta(method),
        }

    def warm_state(self):
        return dict(self.calls)

    def run_guidance(self, request):
        """Method A. Timed from request start to the finished, classified string; includes grammar construction."""
        start = time.perf_counter()
        chat = self.chat_text(config.BASE_PROMPT_TEMPLATE.format(request=request))
        out = methods.run_guidance(self.model, self.tokenizer, chat)
        outcome = methods.classify_html(out["text"], out["hit_cap"])
        seconds = time.perf_counter() - start
        html = out["text"] if outcome[0] == "valid" else None
        return self.record("guidance", outcome, html, out, seconds)

    def run_coercion(self, request):
        """Method B. Timed the same way; includes parsing the JSON and assembling the HTML."""
        start = time.perf_counter()
        text, new_tokens, hit_cap = self.complete(methods.build_prompt(request))
        outcome = methods.classify_json(text, hit_cap)
        seconds = time.perf_counter() - start
        html = methods.json_to_html(text) if outcome[0] == "valid" else None
        result = {"text": text, "new_tokens": new_tokens, "hit_cap": hit_cap}
        return self.record("coercion", outcome, html, result, seconds)

    @modal.method()
    def run_all(self, name, count, overwrite=False):
        """The whole benchmark inside the container, saved to the results volume. Fetch it with `make fetch`."""
        results_volume.reload()  # see runs saved by earlier containers before checking for overwriting
        path = benchmark.results_path(RESULTS_DIR, name, overwrite)
        runners = {"guidance": self.run_guidance, "coercion": self.run_coercion}
        result = benchmark.run_benchmark(
            runners, self.warm_state, config.PROMPTS[:count]
        )
        saved = benchmark.save_results(path, {"probe": self.probe(), **result})
        results_volume.commit()
        return {"path": saved, "records": len(result["records"])}


@app.local_entrypoint()
def smoke():
    """Informal check on whichever GPU is free: both methods on 2 prompts, run inside the container."""
    print(Generator().run_all.remote("smoke", 2, True))


@app.local_entrypoint()
def run(name: str, count: int = 0):
    """The benchmark: the first `count` prompts (0 means all) through both methods, run inside the container
    so a sleeping laptop cannot stop it."""
    print(Generator().run_all.remote(name, count or len(config.PROMPTS), False))
