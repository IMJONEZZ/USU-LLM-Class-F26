import importlib.metadata
import json
import math
import os
import platform
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from src import config, data

REQUIRED_RESULT_FIELDS = (
    "model_name",
    "test_size",
    "batch_size",
    "max_seq_length",
    "versions",
    "metrics",
    "eval_seconds",
    "gpu_summary",
)
_VERSIONED_PACKAGES = ("torch", "transformers", "unsloth", "tokenizers", "peft", "trl")
# a named tuple keeps the formatter from dropping the parentheses (3.14-only syntax)
_SMI_ERRORS = (OSError, subprocess.SubprocessError)


def _now():
    return time.perf_counter()


def cut_at_final_answer(text):
    return data.cut_after_final_answer_line(text)


def _batched(items, size):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def make_generate_fn(model, tokenizer, max_new_tokens=None, batch_size=None):
    max_new_tokens = (
        config.EVAL_MAX_NEW_TOKENS if max_new_tokens is None else max_new_tokens
    )
    batch_size = config.EVAL_BATCH_SIZE if batch_size is None else batch_size

    def generate(prompts):
        if not prompts:
            return []
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        original_side = getattr(tokenizer, "padding_side", "right")
        tokenizer.padding_side = "left"
        outputs = []
        try:
            for batch in _batched(list(prompts), batch_size):
                inputs = tokenizer(batch, return_tensors="pt", padding=True)
                if hasattr(inputs, "to"):
                    inputs = inputs.to(model.device)
                prompt_width = inputs["input_ids"].shape[1]
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                texts = tokenizer.batch_decode(
                    generated[:, prompt_width:], skip_special_tokens=True
                )
                outputs.extend(cut_at_final_answer(text) for text in texts)
        finally:
            tokenizer.padding_side = original_side
        return outputs

    return generate


def grade(row, output):
    final = data.extract_final_answer(output)
    try:
        data.check_chain(output.strip())
        format_ok = True
    except ValueError:
        format_ok = False
    return {
        "id": row["id"],
        "source_id": row.get("source_id", row["id"]),
        "category": row["category"],
        "output": output,
        "final_answer": final,
        "correct": data.is_correct(final, row["answer"], row["accepted_alternatives"]),
        "format_ok": format_ok,
        "length": len(output),
    }


def _metrics(graded):
    per_category = {}
    for category in config.CATEGORIES:
        mine = [g for g in graded if g["category"] == category]
        correct = sum(g["correct"] for g in mine)
        per_category[category] = {
            "n": len(mine),
            "correct": correct,
            "accuracy": correct / len(mine) if mine else "undefined",
        }
    if not graded:
        return {
            "n": 0,
            "accuracy": "undefined",
            "format_compliance": "undefined",
            "avg_response_chars": "undefined",
            "per_category": per_category,
        }
    return {
        "n": len(graded),
        "accuracy": sum(g["correct"] for g in graded) / len(graded),
        "format_compliance": sum(g["format_ok"] for g in graded) / len(graded),
        "avg_response_chars": sum(g["length"] for g in graded) / len(graded),
        "per_category": per_category,
    }


def _grade_all(rows, outputs):
    if len(rows) != len(outputs):
        raise ValueError(f"{len(rows)} rows but {len(outputs)} outputs")
    for row in rows:
        if row["category"] not in config.CATEGORIES:
            raise ValueError(f"category {row['category']!r} is not in CATEGORIES")
    return [grade(row, output) for row, output in zip(rows, outputs, strict=True)]


def score(rows, outputs):
    return _metrics(_grade_all(rows, outputs))


def evaluate(rows, generate_fn, make_prompt):
    outputs = generate_fn([make_prompt(row) for row in rows]) if rows else []
    graded = _grade_all(rows, outputs)
    return graded, _metrics(graded)


def baseline_prompt(row):
    return data.format_prompt(
        f"{config.CHAIN_INSTRUCTIONS}\n\nQuestion: {row['question']}"
    )


def finetuned_prompt(row):
    return data.format_prompt(row["question"])


def check_prompt_length(prompts, tokenizer, limit):
    lengths = [len(data.encode_ids(tokenizer, prompt)) for prompt in prompts]
    over = [n for n in lengths if n > limit]
    if over:
        plural = "prompt" if len(over) == 1 else "prompts"
        raise ValueError(
            f"{len(over)} {plural} longer than {limit} tokens (longest {max(over)})"
        )


def find_weak_categories(accuracy, source_counts, gap, min_sources=None):
    if gap <= 0:
        raise ValueError(f"gap must be positive, got {gap}")
    minimum = config.WEAK_MIN_SOURCES if min_sources is None else min_sources
    eligible = {}
    for category, value in accuracy.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isnan(value) or source_counts.get(category, 0) < minimum:
            continue
        eligible[category] = value
    if len(eligible) < 2:
        return []
    reference = sum(eligible.values()) / len(eligible)
    weak = [c for c, value in eligible.items() if reference - value >= gap - 1e-9]
    return sorted(weak, key=lambda c: (eligible[c], c))


class Timer:
    def __enter__(self):
        self._start = _now()
        self.seconds = None
        return self

    def __exit__(self, *exc):
        self.seconds = _now() - self._start
        return False


def package_versions(names=_VERSIONED_PACKAGES):
    versions = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _nvidia_smi():  # pragma: no cover - needs a GPU machine
    return subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _torch_peak_mib():  # pragma: no cover - needs a GPU machine
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 2**20


class GpuMonitor:
    def __init__(self, run_smi=None, torch_peak_mib=None, interval=2.0):
        self._run_smi = run_smi or _nvidia_smi
        self._torch_peak_mib = torch_peak_mib or _torch_peak_mib
        self._interval = interval
        self._samples = []
        self._stop = threading.Event()
        self._thread = None

    def sample_once(self):
        try:
            text = self._run_smi()
        except _SMI_ERRORS:
            return
        lines = (text or "").strip().splitlines()
        parts = [p.strip() for p in lines[0].split(",")] if lines else []
        if len(parts) != 3:
            return
        try:
            self._samples.append((parts[0], int(float(parts[1])), float(parts[2])))
        except ValueError:
            return

    def _poll(self):
        while not self._stop.wait(self._interval):
            self.sample_once()

    def __enter__(self):
        self.sample_once()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.sample_once()
        return False

    def summary(self):
        if not self._samples:
            return {
                "gpu_name": None,
                "peak_memory_mib": None,
                "mean_utilization_percent": None,
                "peak_torch_memory_allocated_mib": self._torch_peak_mib(),
            }
        return {
            "gpu_name": self._samples[0][0],
            "peak_memory_mib": max(s[1] for s in self._samples),
            "mean_utilization_percent": sum(s[2] for s in self._samples)
            / len(self._samples),
            "peak_torch_memory_allocated_mib": self._torch_peak_mib(),
        }


def build_result(
    *,
    split,
    metrics,
    rows,
    batch_size,
    max_seq_length,
    versions,
    eval_seconds,
    gpu_summary,
    prompt_style,
    max_new_tokens,
    model_name=None,
):
    return {
        "model_name": model_name or config.MODEL_NAME,
        "split": split,
        "test_size": len(rows),
        "batch_size": batch_size,
        "max_seq_length": max_seq_length,
        "versions": versions,
        "metrics": metrics,
        "eval_seconds": eval_seconds,
        "gpu_summary": gpu_summary,
        "gpu_type": config.GPU_TYPE,
        "prompt_style": prompt_style,
        "max_new_tokens": max_new_tokens,
    }


def save_result(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def ensure_split_allowed(split, result_path, force):
    if split not in ("eval", "test"):
        raise ValueError(f"split must be 'eval' or 'test', got {split!r}")
    if split == "test" and Path(result_path).exists() and not force:
        raise FileExistsError(
            f"{result_path} exists: the final test set is run once; pass --force to "
            "run it again"
        )


def default_output_path(split, prompt_style, output=""):
    if output:
        return Path(output)
    if split == "test" and prompt_style == "finetuned":
        return config.FINAL_TEST_PATH
    if split == "eval" and prompt_style == "baseline":
        return config.BASELINE_PATH
    raise ValueError(
        f"no default result file for split={split!r} with {prompt_style!r}; "
        "pass --output"
    )


def load_split_rows(split):
    if split == "eval":
        rows = data.read_jsonl(config.EVAL_PATH)
        if not rows:
            raise ValueError(f"no rows in {config.EVAL_PATH}")
        return rows
    if split == "test":
        return data.load_test_set()
    raise ValueError(f"split must be 'eval' or 'test', got {split!r}")
