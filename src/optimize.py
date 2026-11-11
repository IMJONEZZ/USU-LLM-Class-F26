import hashlib
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
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


def write_baseline(lm, tasks, splits, path, search=False):
    """Ask Gemini for the baseline instruction and save it beside whatever else the file holds.

    Uses the first train task of each type. Refuses to overwrite a baseline that is already saved.
    """
    saved = json.loads(Path(path).read_text()) if Path(path).exists() else {}
    if "baseline" in saved:
        raise FileExistsError(
            f"{path} already holds a baseline; it is frozen once written"
        )
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
    tools = (
        {"tools": [{"googleSearch": {}}]} if search else {}
    )  # Gemini's built-in Google Search
    reply = lm(messages=[{"role": "user", "content": meta_prompt}], **tools)[0]
    reply = (
        reply["text"] if isinstance(reply, dict) else reply
    )  # with tools, DSPy returns {"text": ..., "tool_calls": ...}
    text, _, source_lines = reply.partition(
        "---SOURCES---"
    )  # the writer lists any pages it used after this line
    baseline = text.strip()
    if not baseline:
        raise ValueError("the writer returned an empty instruction")
    saved["baseline"] = baseline
    saved["writer"] = {
        "model": config.GEMINI_MODEL,
        "search_requested": search,
        "sources_used": [
            line.strip() for line in source_lines.splitlines() if line.strip()
        ],
        "guideline_sources": prompts.WRITER_GUIDELINE_SOURCES,
        "meta_prompt": meta_prompt,
        "meta_prompt_sha256": hashlib.sha256(meta_prompt.encode()).hexdigest(),
        "example_keys": [t["key"] for t in chosen],
    }
    Path(path).write_text(json.dumps(saved, indent=2))
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
    # Gemini 3 models spend much of max_tokens on hidden thinking, and thinking effort is raised, so the limit is generous
    lm = dspy.LM(
        config.GEMINI_MODEL,
        api_key=os.environ["GOOGLE_API_KEY"],
        max_tokens=32000,
        reasoning_effort="high",
    )
    try:
        text = write_baseline(lm, tasks, splits, config.PROMPTS_PATH, search=True)
    except dspy.utils.exceptions.LMError as error:
        print(
            f"the call with Google Search failed ({type(error).__name__}: {str(error)[:200]}); retrying once without search"
        )
        text = write_baseline(lm, tasks, splits, config.PROMPTS_PATH, search=False)
    saved = json.loads(Path(config.PROMPTS_PATH).read_text())
    last = lm.history[-1]
    print(f"baseline instruction ({len(text.split())} words):\n\n{text}\n")
    print(
        f"search requested: {saved['writer']['search_requested']}; sources the writer listed: {saved['writer']['sources_used']}"
    )
    print(
        f"actual: {len(lm.history)} call(s), usage {last['usage']}, cost ${last['cost']} (search grounding fees, if any, are not in this figure)"
    )


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


def smoke_record_from_log(log_text, candidates):
    """Rebuild the smoke run's record (same shape as results/gepa.json) from its printed log.

    The smoke run was made before records were saved, so its numbers are read back from the log.
    Gemini's token counts were never printed, so they are None.
    """
    summary = re.search(
        r"candidates (\d+), validation scores \[([^\]]*)\], best (\d+), metric calls (\d+)",
        log_text,
    )
    actual = re.search(
        r"actual: (\d+) student calls, (\d+) Gemini calls, Gemini cost \$([0-9.]+)",
        log_text,
    )
    stamps = re.findall(
        r"^(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})", log_text, flags=re.MULTILINE
    )
    # the log's own clock has no timezone; both stamps are read as UTC, which leaves their difference unchanged
    first, last = (
        datetime.strptime(s, "%Y/%m/%d %H:%M:%S").replace(tzinfo=UTC)
        for s in (stamps[0], stamps[-1])
    )
    seconds = (last - first).total_seconds()
    scores = [float(s) for s in summary[2].split(",")]
    return {
        "seed": config.SEED,
        "source": "rebuilt from results/logs/m6_smoke.txt and the GEPA candidates file; Gemini tokens were not recorded",
        "total_metric_calls": int(summary[4]),
        "best_index": int(summary[3]),
        "candidates": [
            {"index": i, "validation_score": round(score, 2), "instruction": c["self"]}
            for i, (score, c) in enumerate(zip(scores, candidates))
        ],
        "student": {
            "model": config.SERVED_NAME,
            "calls": int(actual[1]),
            "seconds": round(seconds, 2),
            "dollars": round(seconds * config.T4_PER_SECOND, 2),
        },
        "gemini": {
            "model": config.GEMINI_MODEL,
            "calls": int(actual[2]),
            "prompt_tokens": None,
            "completion_tokens": None,
            "dollars": round(float(actual[3]), 2),
        },
    }


def smoke_record(
    log_path="results/logs/m6_smoke.txt", gepa_dir="results/gepa_smoke_200528"
):
    candidates = json.loads((Path(gepa_dir) / "candidates.json").read_text())
    record = smoke_record_from_log(Path(log_path).read_text(), candidates)
    with (Path(config.RESULTS_DIR) / "gepa_smoke.json").open(
        "x"
    ) as f:  # "x": never overwrite
        json.dump(record, f, indent=2)
    print(
        f"wrote {config.RESULTS_DIR}/gepa_smoke.json: scores {[c['validation_score'] for c in record['candidates']]}, student {record['student']}, gemini {record['gemini']}"
    )


def main(step="demo"):
    {
        "demo": demo,
        "baseline": baseline,
        "gepa-smoke": gepa_smoke,
        "gepa": gepa_full,
        "smoke-record": smoke_record,
    }[step]()


if __name__ == "__main__":
    main(sys.argv[1])
