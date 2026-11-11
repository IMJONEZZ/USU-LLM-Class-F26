"""A10 workflow: unconstrained DSPy optimization, held-out evaluation, and reporting."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import os
import time
from pathlib import Path
from statistics import mean

os.environ.setdefault("DSPY_CACHEDIR", "/tmp/a10-dspy-cache")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import dspy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
from dspy.utils.exceptions import AdapterParseError

from src.html_evaluation import evaluate_html, metric

DATASET = Path("fixtures/a10.json")
OUTPUT = Path("artifacts/a10")
DEFAULT_CONFIG = {
    "model": "qwen2.5:1.5b",
    "api_base": "http://localhost:11434",
    "temperature": 0.5,
    "max_tokens": 1024,
    "num_ctx": 8192,
    "num_gpu": 99,
    "seeds": [11, 22, 33],
    "structural_weight": 0.5,
    "instruction_weight": 0.5,
    "grammar_policy": "Match returned HTML exactly; boundary-newline diagnostic does not affect scores.",
    "optimizer": "MIPROv2 auto=light, max_bootstrapped_demos=2, max_labeled_demos=0, metric_threshold=1.0, num_threads=1",
    "score_version": "v3-document-order-and-list-item-order-audited",
}

BASELINE_INSTRUCTIONS = """Generate a concise complete HTML document following the request.
Use only html, head, title, body, h1, h2, p, ul, ol, li, strong, em, br.
Do not use attributes, a doctype, comments, markdown fences, or explanatory text.
Use this structure: html contains head (with title), then body. Body contains
1-6 headings, paragraphs or lists. Lists contain 1-4 li elements. Headings,
titles and list items contain plain text. Paragraphs contain plain text and
optionally strong, em or <br />. Put each block tag on its own line, while
keeping title, headings, paragraphs and li text on the same line as their tags.
Use a newline after each block, including the final </html>."""


class HtmlPage(dspy.Signature):
    request: str = dspy.InputField()
    html: str = dspy.OutputField(desc="The complete requested HTML document.")


class ProposalRetryAdapter(dspy.ChatAdapter):
    """Retry malformed optimizer instructions, without retrying scored HTML."""

    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        # The small proposal model can echo inputs instead of its output field.
        # Retry only that internal proposal step; evaluation still gets one call.
        attempts = 3 if set(signature.output_fields) == {"proposed_instruction"} else 1
        for attempt in range(attempts):
            try:
                return super().__call__(lm, lm_kwargs, signature, demos, inputs)
            except AdapterParseError:
                if attempt == attempts - 1:
                    raise
                logging.getLogger(__name__).warning(
                    "Malformed MIPROv2 instruction proposal; retrying (attempt %s/%s).",
                    attempt + 2,
                    attempts,
                )


def program():
    return dspy.Predict(HtmlPage.with_instructions(BASELINE_INSTRUCTIONS))


def read_inputs():
    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET}. Supply --dataset PATH; see README for the JSON format."
        )
    bundle = json.loads(DATASET.read_text())
    rows, config = bundle["examples"], DEFAULT_CONFIG | bundle.get("config", {})
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Dataset IDs must be unique")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Supply at least one seed, without duplicates")
    # Keep expected answers available to the metric, but expose only the request
    # as a predictor input. The test split is never passed to MIPROv2.
    splits = {
        split: [
            dspy.Example(**row).with_inputs("request")
            for row in rows
            if row["split"] == split
        ]
        for split in ("train", "validation", "test")
    }
    if any(not examples for examples in splits.values()):
        raise ValueError("Dataset needs nonempty train, validation and test splits")
    config.setdefault("challenge_id", splits["test"][0].id)
    if config["challenge_id"] not in {example.id for example in splits["test"]}:
        raise ValueError("challenge_id must identify a test request")
    return rows, splits, config


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def require_gpu(config):
    """Load only (no prompt), then reject partial/CPU placement before inference."""
    base = config["api_base"]
    response = requests.post(
        base + "/api/generate",
        json={
            "model": config["model"],
            "stream": False,
            "keep_alive": "60m",
            "options": {"num_gpu": config["num_gpu"], "num_ctx": config["num_ctx"]},
        },
        timeout=120,
    )
    response.raise_for_status()
    response = requests.get(base + "/api/ps", timeout=10)
    response.raise_for_status()
    loaded = next(m for m in response.json()["models"] if m["name"] == config["model"])
    if loaded["size_vram"] < loaded["size"]:
        raise RuntimeError("Full GPU offload required. No CPU inference is permitted.")
    return loaded


def configure(config):
    import litellm
    from litellm.integrations.custom_logger import CustomLogger

    class RequestAudit(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            # Inspect the outgoing provider payload to verify that seeds actually
            # reach Ollama and no structured-output format was enabled.
            payload = kwargs["additional_args"]["complete_input_dict"]
            lm.api_requests.append(
                {
                    "options": payload.get("options", {}),
                    "format": payload.get("format"),
                    "timestamp_ns": time.time_ns(),
                }
            )

    lm = dspy.LM(
        "ollama_chat/" + config["model"],
        api_base=config["api_base"],
        engine="litellm",
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        num_ctx=config["num_ctx"],
        num_gpu=config["num_gpu"],
        keep_alive="60m",
        cache=False,
        num_retries=0,
    )
    lm.api_requests = []
    lm.audit_logger = RequestAudit()
    litellm.callbacks = [lm.audit_logger]
    # A JSON fallback would change the decoding conditions after a parse failure.
    dspy.configure(lm=lm, adapter=ProposalRetryAdapter(use_json_adapter_fallback=False))
    return lm


def predict_once(predictor, example, lm, seed):
    """No output repair or retry. Preserve adapter failures and underlying response."""
    before = len(lm.history)
    before_api = len(lm.api_requests)
    started = time.monotonic()
    error = None
    try:
        prediction = predictor(request=example.request, config={"seed": seed})
        html = prediction.html
    except AdapterParseError as exc:
        # Preserve the raw completion below, but score an unusable HTML field as
        # a failure instead of repairing it or sampling again.
        html = ""
        error = str(exc)
    history = lm.history[before:]
    if len(history) != 1:
        raise RuntimeError(
            f"Expected one fresh inference call, recorded {len(history)}"
        )
    call = history[0]
    api_calls = lm.api_requests[before_api:]
    if len(api_calls) != 1 or api_calls[0]["options"].get("seed") != seed:
        raise RuntimeError("Expected one fresh Ollama request with the requested seed")
    if api_calls[0]["format"]:
        raise RuntimeError("Constrained output format must not be enabled")
    result = evaluate_html(html, example.requirements)
    if error:
        result = {
            **result,
            "score": 0.0,
            "instruction_score": 0.0,
            "instruction_pass": False,
        }
    return {
        "request_id": example.id,
        "request": example.request,
        "seed": seed,
        "html": html,
        "raw_response": call["outputs"],
        "error": error,
        "elapsed_seconds": time.monotonic() - started,
        "lm_kwargs": call["kwargs"],
        "usage": call.get("usage"),
        "api_request": api_calls[0],
        **result,
    }


def smoke(splits, config):
    gpu = require_gpu(config)
    lm = configure(config)
    result = predict_once(program(), splits["train"][0], lm, 11)
    if result["error"]:
        raise RuntimeError(
            "Baseline DSPy field could not be parsed: " + result["error"]
        )
    proposer = dspy.Predict("task -> instruction")
    proposal = proposer(
        task="Suggest one concise instruction for generating HTML without markdown."
    )
    if not proposal.instruction.strip():
        raise RuntimeError("Instruction proposal was empty")
    write_json(
        OUTPUT / "smoke.json",
        {"gpu": gpu, "prediction": result, "proposal": proposal.instruction},
    )
    print("GPU smoke test passed; score:", result["score"], flush=True)


def optimize(splits, config):
    gpu = require_gpu(config)
    lm = configure(config)
    baseline = program()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    baseline.save(OUTPUT / "baseline.json")
    optimizer = dspy.MIPROv2(
        metric=metric,
        prompt_model=lm,
        task_model=lm,
        auto="light",
        max_bootstrapped_demos=2,
        max_labeled_demos=0,
        # Partial scores guide search; only fully passing outputs become demos.
        metric_threshold=1.0,
        num_threads=1,
        log_dir=str(OUTPUT / "optimizer_logs"),
    )
    started = time.monotonic()
    optimized = optimizer.compile(
        # Compilation must not mutate the baseline used in the final comparison.
        baseline.deepcopy(),
        trainset=splits["train"],
        valset=splits["validation"],
        minibatch=False,
    )
    optimized.save(OUTPUT / "optimized.json")
    restored = program()
    restored.load(OUTPUT / "optimized.json")
    assert restored.signature.instructions == optimized.signature.instructions
    assert restored.dump_state()["demos"] == optimized.dump_state()["demos"]
    write_json(
        OUTPUT / "optimization.json",
        {
            "gpu": gpu,
            "elapsed_seconds": time.monotonic() - started,
            "lm_calls": len(lm.history),
            "demonstrations": len(optimized.demos),
            "best_validation_score": getattr(optimized, "score", None),
            "dspy_version": dspy.__version__,
            "config": config,
            "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        },
    )
    export_prompts(baseline, restored)
    print(
        "Optimization saved and reloaded; demonstrations:",
        len(restored.demos),
        flush=True,
    )


def export_prompts(baseline, optimized):
    text = "# Assignment 10 prompts\n\n"
    for name, predictor in (("Baseline", baseline), ("Optimized", optimized)):
        text += f"## {name}\n\n```text\n{predictor.signature.instructions}\n```\n\n"
        text += (
            "Demonstrations:\n\n```json\n"
            + json.dumps(predictor.dump_state()["demos"], indent=2)
            + "\n```\n\n"
        )
    (OUTPUT / "prompts.md").write_text(text)


def evaluate(splits, config):
    snapshot = OUTPUT / "evaluation_config.json"
    if snapshot.exists():
        # Resuming under different settings would mix incompatible measurements.
        previous = json.loads(snapshot.read_text())
        unchanged = all(previous[key] == value for key, value in config.items())
        unchanged &= (
            previous["dataset_sha256"]
            == hashlib.sha256(DATASET.read_bytes()).hexdigest()
        )
        if not unchanged:
            raise ValueError(
                "Evaluation inputs changed; archive the previous results first"
            )
    gpu = require_gpu(config)
    lm = configure(config)
    write_json(
        OUTPUT / "evaluation_config.json",
        {
            **config,
            "gpu": gpu,
            "dspy_version": dspy.__version__,
            "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        },
    )
    write_json(OUTPUT / "held_out.json", [dict(row) for row in splits["test"]])
    path = OUTPUT / "responses.jsonl"
    completed = (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )
    keys = {(row["program"], row["request_id"], row["seed"]) for row in completed}
    for name in ("baseline", "optimized"):
        predictor = program()
        predictor.load(OUTPUT / f"{name}.json")
        for example in splits["test"]:
            for seed in config["seeds"]:
                if (name, example.id, seed) in keys:
                    continue
                result = {"program": name, **predict_once(predictor, example, lm, seed)}
                with path.open("a") as stream:
                    # Save each sample immediately so an interrupted run can resume.
                    stream.write(json.dumps(result, default=str) + "\n")
                print(name, example.id, seed, round(result["score"], 3), flush=True)
    # Verify offload remained complete after all inference requests.
    require_gpu(config)


def main():
    global DATASET
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["smoke", "optimize", "evaluate", "report"])
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET,
        help="Local HTML request dataset JSON (see README)",
    )
    args = parser.parse_args()
    DATASET = args.dataset
    try:
        _, splits, config = read_inputs()
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    if args.command == "report":
        reference = Path("artifacts/a9_reference")
        report(OUTPUT, config, reference=reference if reference.exists() else None)
    else:
        {"smoke": smoke, "optimize": optimize, "evaluate": evaluate}[args.command](
            splits, config
        )


def report(output, config, reference=None):
    rows = [
        json.loads(line)
        for line in (output / "responses.jsonl").read_text().splitlines()
    ]
    keys = {(r["program"], r["request_id"], r["seed"]) for r in rows}
    expected = {
        (name, r["id"], seed)
        for name in ("baseline", "optimized")
        for r in json.loads((output / "held_out.json").read_text())
        for seed in config["seeds"]
    }
    if keys != expected or len(rows) != len(expected):
        # Missing or duplicated samples would distort the before/after comparison.
        raise ValueError(f"Expected exactly {len(expected)} unique evaluation records")
    programs = ["baseline", "optimized"]
    if reference is not None:
        # A9 is loaded only for reporting; it never supplies optimization examples.
        reference_rows = [
            json.loads(line)
            for line in (reference / "responses.jsonl").read_text().splitlines()
        ]
        expected_reference = {
            ("a9_reference", r["id"], seed)
            for r in json.loads((output / "held_out.json").read_text())
            for seed in config["seeds"]
        }
        actual_reference = {
            (r["program"], r["request_id"], r["seed"]) for r in reference_rows
        }
        if actual_reference != expected_reference or len(reference_rows) != len(
            expected_reference
        ):
            raise ValueError(
                f"Expected exactly {len(expected_reference)} unique A9 reference records"
            )
        rows = reference_rows + rows
        programs.insert(0, "a9_reference")
    metrics = [
        ("score", "Composite score"),
        ("tag_nesting_valid", "Tag/nesting validity"),
        ("grammar_pass", "Exact grammar"),
        ("instruction_pass", "Instruction pass"),
    ]
    summary = {}
    text = "# Assignment 10 measured results\n\n| Program | Mean score | Tag/nesting validity | Exact grammar | Instruction pass |\n|---|---:|---:|---:|---:|\n"
    for name in programs:
        group = [r for r in rows if r["program"] == name]
        summary[name] = {key: mean(r[key] for r in group) for key, _ in metrics}
        values = summary[name]
        text += f"| {name} | {values['score']:.3f}"
        for key in ("tag_nesting_valid", "grammar_pass", "instruction_pass"):
            text += f" | {sum(r[key] for r in group)}/{len(group)} ({values[key]:.1%})"
        text += " |\n"
    text += f"\n{len(expected) // (2 * len(config['seeds']))} held-out requests, {len(config['seeds'])} fresh generations per request. Repeated samples are grouped by request; this is a small classroom experiment.\n"
    text += "\nExact grammar includes internal whitespace and the terminal newline. DSPy's ChatAdapter strips boundary whitespace; boundary-newline diagnostic matches are recorded separately in raw results and do not count as exact passes.\n"
    if reference is not None:
        text += "\nThe primary experiment is A10 baseline versus A10 optimized. A9 is a reference: it uses Guidance/Transformers 4-bit GPU inference and the original grammar/local generation limits, rather than Ollama GGUF and its 1,024-token output budget. Its outputs were generated after optimization and never supplied to MIPROv2. All three groups are scored with the same evaluator.\n"
    (output / "results.md").write_text(text)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), layout="constrained")
    labels = {
        "a9_reference": "A9\nreference",
        "baseline": "A10\nbaseline",
        "optimized": "A10\noptimized",
    }
    colors = {"a9_reference": "#b45309", "baseline": "#64748b", "optimized": "#0f766e"}
    for ax, (key, label) in zip(axes, metrics, strict=True):
        values = [summary[name][key] for name in programs]
        bars = ax.bar(
            [labels[n] for n in programs], values, color=[colors[n] for n in programs]
        )
        ax.set(ylim=(0, 1.13), title=label, yticks=[0, 0.25, 0.5, 0.75, 1])
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=3)
    fig.suptitle(
        "HTML generation: A10 prompt optimization with A9 constrained reference"
        if reference
        else "A10 unconstrained HTML prompt optimization"
    )
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)
    # This illustrative seed was selected after evaluation. The caption discloses
    # that choice, while the aggregate table and graph include every sample.
    challenge = [
        r
        for r in rows
        if r["program"] in ("baseline", "optimized")
        and r["request_id"] == config["challenge_id"]
        and r["seed"] == config["seeds"][min(1, len(config["seeds"]) - 1)]
    ]
    page = '<!doctype html><meta charset="utf-8"><title>A10 before and after</title><style>body{font:16px system-ui;margin:32px;background:#f8fafc;color:#172033}main{display:grid;grid-template-columns:1fr 1fr;gap:24px}section{background:white;padding:20px;border:1px solid #cbd5e1}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px}h1{margin-bottom:8px}</style><h1>Assignment 10: before and after</h1>'
    page += f"<p>Same request, model, temperature, token budget, and seed. Request selected before evaluation. This seed was selected afterward to illustrate a response-framing improvement; the results table includes all {len(expected)} A10 samples.</p>"
    page += "<p>" + html.escape(challenge[0]["request"]) + "</p><main>"
    for row in challenge:
        # Show generated markup as escaped source rather than executing it.
        page += f"<section><h2>{html.escape(row['program'].title())}</h2><p>Score: {row['score']:.3f} · Grammar: {row['grammar_pass']} · Instructions: {row['instruction_pass']}</p><pre>{html.escape(row['html'] or str(row['raw_response']))}</pre></section>"
    (output / "before_after.html").write_text(page + "</main>")
    print(text)


if __name__ == "__main__":
    main()
