import hashlib
import json
import os
import sys
import time
from pathlib import Path

import dspy

from src import checks, config, data, estimate, prompts


def make_student(instruction=prompts.ONE_LINER):
    """The student LM and the program that uses it with the given instruction; every run that calls the server starts here."""
    lm = dspy.LM(
        "openai/"
        + config.SERVED_NAME,  # "openai/" = speak the OpenAI HTTP format, which vLLM also speaks
        api_base=os.environ["MODAL_URL"] + "/v1",
        api_key="unused",  # litellm refuses an empty key; vLLM ignores it and Modal's proxy tokens in headers do the checking
        headers={
            "Modal-Key": os.environ["MODAL_ID"],
            "Modal-Secret": os.environ["MODAL_SECRET"],
        },
        model_type="chat",
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_NEW_TOKENS,
        cache=False,
    )
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False))

    # a DSPy module: turns the signature into a prompt, calls the LM, and parses the reply into fields
    predict = dspy.Predict(dspy.Signature(prompts.SIGNATURE, instruction))
    return lm, predict


def demo():
    # about 3 short calls to the student model; the seconds figure is a guess until M4 measures real call times
    estimate.estimate(
        [{"model": config.SERVED_NAME, "calls": 3, "seconds": 30, "basis": "assumed"}]
    )
    lm, predict = make_student()

    splits = json.loads(Path(config.SPLITS_PATH).read_text())
    _, tasks = data.load_examples()
    # a task is one IFEval prompt plus the rule its answer must follow; this one is from train, so test stays untouched until M8
    task = next(t for t in tasks if t["key"] == splits["train"][0])

    start = time.time()
    replies = []
    for _ in range(2):
        try:
            replies.append(predict(prompt=task["prompt"]).response)
        except dspy.utils.exceptions.AdapterParseError as error:
            # the model answered without DSPy's [[ ## response ## ]] marker
            replies.append("PARSE ERROR, raw reply: " + error.lm_response)
    seconds = time.time() - start

    print("prompt:", task["prompt"][:100])
    for number, reply in enumerate(replies, 1):
        print(f"reply {number}: {reply[:200]!r}")
    print("replies differ:", replies[0] != replies[1])

    print("\nthe messages DSPy sent for the last call:")
    for message in lm.history[-1]["messages"]:
        print(f"[{message['role']}]\n{message['content']}\n")

    output_tokens = sum(h["usage"]["completion_tokens"] for h in lm.history)
    print(
        f"actual: {len(lm.history)} calls, {seconds:.0f} s "
        f"(about ${seconds * config.T4_PER_SECOND:.2f}), {output_tokens} output tokens"
    )


def write_baseline(lm, tasks, splits, path):
    """Ask Gemini for the baseline instruction and save it. Uses the first train task of each type; refuses to overwrite."""
    train = set(splits["train"])
    first = {}
    for task in sorted(tasks, key=lambda t: t["key"]):
        if task["key"] in train:
            first.setdefault(task["instruction_id"], task)
    chosen = [first[kind] for kind in sorted(first)]
    samples = "\n\n".join(
        f"[{n}] Rule ({t['instruction_id']}): {checks.describe(t)}\nTask prompt: {t['prompt']}"
        for n, t in enumerate(chosen, 1)
    )
    meta_prompt = prompts.WRITER_META_PROMPT.format(samples=samples)
    baseline = lm(messages=[{"role": "user", "content": meta_prompt}])[0].strip()
    if not baseline:
        raise ValueError("the writer returned an empty instruction")
    record = {
        "baseline": baseline,
        "writer": {
            "model": config.GEMINI_MODEL,
            "meta_prompt": meta_prompt,
            "meta_prompt_sha256": hashlib.sha256(meta_prompt.encode()).hexdigest(),
            "example_keys": [t["key"] for t in chosen],
        },
    }
    with Path(path).open("x") as f:  # "x": the baseline is frozen once written
        json.dump(record, f, indent=2)
    return baseline


def baseline():
    splits = json.loads(Path(config.SPLITS_PATH).read_text())
    _, tasks = data.load_examples()
    # 24 train prompts plus the writer prompt in, one short instruction (plus hidden thinking) out
    estimate.estimate(
        [
            {
                "model": config.GEMINI_MODEL,
                "calls": 1,
                "input_tokens": 2000,
                "output_tokens": 2000,
                "basis": "assumed",
            }
        ]
    )
    # Gemini 3 models spend part of max_tokens on hidden thinking, so the limit is generous
    lm = dspy.LM(
        config.GEMINI_MODEL, api_key=os.environ["GOOGLE_API_KEY"], max_tokens=8000
    )
    text = write_baseline(lm, tasks, splits, config.PROMPTS_PATH)
    last = lm.history[-1]
    print(f"baseline instruction ({len(text.split())} words):\n\n{text}\n")
    print(f"actual: 1 call, usage {last['usage']}, cost ${last['cost']}")


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """GEPA's score and feedback: 1.0 when Google's strict check passes the parsed `response` field, else 0.0 with feedback."""
    feedback = checks.feedback(gold.task, getattr(pred, "response", None) or "")
    if feedback is None:
        return dspy.Prediction(
            score=1.0, feedback="The response followed the instruction."
        )
    return dspy.Prediction(score=0.0, feedback=feedback)


def gepa_examples(tasks, keys):
    """DSPy examples for the tasks with these keys; the task rides along so the metric can run the checker."""
    wanted = set(keys)
    return [
        dspy.Example(prompt=t["prompt"], task=t).with_inputs("prompt")
        for t in tasks
        if t["key"] in wanted
    ]


def build_gepa(reflection_lm, log_dir, **budget):
    """A GEPA optimizer with exactly one budget setting (auto, max_full_evals or max_metric_calls)."""
    if len(budget) != 1 or not set(budget) <= {
        "auto",
        "max_full_evals",
        "max_metric_calls",
    }:
        raise ValueError(
            f"set exactly one of auto, max_full_evals, max_metric_calls; got {sorted(budget)}"
        )
    return dspy.GEPA(
        metric=metric,
        reflection_lm=reflection_lm,
        add_format_failure_as_feedback=True,  # Madi's decision: Gemini sees the replies that lost their marker
        track_stats=True,
        log_dir=log_dir,
        seed=config.SEED,
        num_threads=config.WORKERS,
        **budget,
    )


def run_gepa(tasks, train_keys, val_keys, log_dir, **budget):
    """Optimize the student's instruction; returns (optimized program, student LM, reflection LM)."""
    student_lm, predict = make_student()
    # Gemini reflects at temperature 1.0 with a large limit because it spends part of it on hidden thinking
    reflection_lm = dspy.LM(
        config.GEMINI_MODEL,
        api_key=os.environ["GOOGLE_API_KEY"],
        temperature=1.0,
        max_tokens=16000,
    )
    gepa = build_gepa(reflection_lm, log_dir, **budget)
    optimized = gepa.compile(
        predict,
        trainset=gepa_examples(tasks, train_keys),
        valset=gepa_examples(tasks, val_keys),
    )
    return optimized, student_lm, reflection_lm


def gepa_smoke():
    splits = json.loads(Path(config.SPLITS_PATH).read_text())
    _, tasks = data.load_examples()
    train_keys, val_keys = (
        splits["train"][:6],
        splits["train"][6:12],
    )  # the smoke run uses train tasks only
    budget = 40  # covers one pass over the 6 validation tasks plus a few rounds of 3-task minibatches
    estimate.estimate(
        [
            {
                "model": config.SERVED_NAME,
                "calls": budget,
                "seconds": budget * 1.6,
                "basis": "assumed",
            },
            {
                "model": config.GEMINI_MODEL,
                "calls": 4,
                "input_tokens": 3000,
                "output_tokens": 3000,
                "basis": "assumed",
            },
        ]
    )
    log_dir = f"{config.RESULTS_DIR}/gepa_smoke_{time.strftime('%H%M%S')}"
    optimized, student_lm, reflection_lm = run_gepa(
        tasks, train_keys, val_keys, log_dir, max_metric_calls=budget
    )

    details = optimized.detailed_results
    scores = [round(s, 2) for s in details.val_aggregate_scores]
    print(
        f"\ncandidates {len(details.candidates)}, validation scores {scores}, best {details.best_idx}, metric calls {details.total_metric_calls}"
    )
    for number, candidate in enumerate(details.candidates):
        print(f"\ncandidate {number}: {candidate.signature.instructions}")
    print("\nwhat Gemini was shown (start of its first reflection prompt):")
    print(reflection_lm.history[0]["messages"][-1]["content"][:1200])
    cost = sum(h["cost"] or 0 for h in reflection_lm.history)
    print(
        f"\nactual: {len(student_lm.history)} student calls, {len(reflection_lm.history)} Gemini calls, Gemini cost ${cost:.4f}"
    )


def summarize_gepa(details, student_lm, reflection_lm, seconds, log_dir):
    """The numbers worth keeping from a GEPA run: candidates and scores, calls, tokens, seconds and dollars by model."""
    usage = [h["usage"] for h in reflection_lm.history]
    return {
        "seed": config.SEED,
        "log_dir": log_dir,
        "total_metric_calls": details.total_metric_calls,
        "full_validation_evals": details.num_full_val_evals,
        "best_index": details.best_idx,
        "candidates": [
            {
                "index": i,
                "validation_score": round(score, 2),
                "instruction": c.signature.instructions,
            }
            for i, (score, c) in enumerate(
                zip(details.val_aggregate_scores, details.candidates)
            )
        ],
        "student": {
            "model": config.SERVED_NAME,
            "calls": len(student_lm.history),
            "seconds": round(seconds, 2),
            "dollars": round(seconds * config.T4_PER_SECOND, 2),
        },
        "gemini": {
            "model": config.GEMINI_MODEL,
            "calls": len(usage),
            "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage),
            "completion_tokens": sum(u.get("completion_tokens", 0) for u in usage),
            "dollars": round(sum(h["cost"] or 0 for h in reflection_lm.history), 2),
        },
    }


def gepa_full():
    splits = json.loads(Path(config.SPLITS_PATH).read_text())
    saved = json.loads(Path(config.PROMPTS_PATH).read_text())
    if "gepa" in saved:
        raise FileExistsError(
            "a GEPA instruction is already saved; keep it and start a new run folder instead"
        )
    _, tasks = data.load_examples()
    calls = estimate.gepa_light_calls(len(splits["validation"]))
    estimate.estimate(
        [
            {
                "model": config.SERVED_NAME,
                "calls": calls,
                "seconds": calls * 1.6,
                "basis": "assumed",
            },
            {
                "model": config.GEMINI_MODEL,
                "calls": 15,
                "input_tokens": 4000,
                "output_tokens": 4000,
                "basis": "assumed",
            },
        ]
    )
    log_dir = f"{config.RESULTS_DIR}/gepa_{time.strftime('%Y%m%d_%H%M%S')}"
    start = time.time()
    optimized, student_lm, reflection_lm = run_gepa(
        tasks, splits["train"], splits["validation"], log_dir, auto="light"
    )
    record = summarize_gepa(
        optimized.detailed_results,
        student_lm,
        reflection_lm,
        time.time() - start,
        log_dir,
    )

    saved["gepa"] = optimized.signature.instructions
    Path(config.PROMPTS_PATH).write_text(json.dumps(saved, indent=2))
    with (Path(config.RESULTS_DIR) / "gepa.json").open(
        "x"
    ) as f:  # "x": never overwrite an earlier run
        json.dump(record, f, indent=2)
    print(f"GEPA instruction:\n\n{saved['gepa']}\n")
    print(
        f"scores {[c['validation_score'] for c in record['candidates']]}, best {record['best_index']}"
    )
    print(f"actual: student {record['student']}, gemini {record['gemini']}")


def main(step="demo"):
    {"demo": demo, "baseline": baseline, "gepa-smoke": gepa_smoke, "gepa": gepa_full}[
        step
    ]()


if __name__ == "__main__":
    main(sys.argv[1])
