import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dspy
import numpy as np
import openai
import scipy.stats
from tqdm import tqdm

from src import checks, config, data, estimate, optimize, prompts

OUTCOMES = ("pass", "fail", "parse_error", "request_failure")


def draw(predict, task):
    """Ask the model once. Returns {"outcome": pass, fail, parse_error or request_failure, "reply": the text}."""
    try:
        reply = predict(prompt=task["prompt"]).response
    except (
        dspy.utils.exceptions.AdapterParseError
    ) as error:  # the reply had no [[ ## response ## ]] marker
        return {"outcome": "parse_error", "reply": error.lm_response}
    except (
        dspy.utils.exceptions.LMError
    ) as error:  # the server or network failed; the model is not to blame
        return {"outcome": "request_failure", "reply": str(error)}
    return {"outcome": "pass" if checks.passes(task, reply) else "fail", "reply": reply}


def acceptance_rate(predict, task, n):
    """Ask the model n times about one task and return how often it passed."""
    all_results = {"pass": 0, "fail": 0, "parse_error": 0, "request_failure": 0}
    for _ in range(n):
        single_result = draw(predict, task)["outcome"]
        all_results[single_result] += 1
    denominator = all_results["pass"] + all_results["fail"] + all_results["parse_error"]
    if denominator > 0:
        all_results["rate"] = all_results["pass"] / denominator
    else:
        all_results["rate"] = None
    return all_results


def pilot_tasks(tasks, splits):
    """Train and validation tasks only, each tagged with its split; there is no way to ask for test from here."""
    split_of = {key: name for name in ("train", "validation") for key in splits[name]}
    return [{**t, "split": split_of[t["key"]]} for t in tasks if t["key"] in split_of]


def run_test_draws():
    """Ask every test task TEST_DRAWS times for each of the three instructions; one saved line per finished task and instruction."""
    splits = json.loads(Path(config.SPLITS_PATH).read_text())
    saved = json.loads(Path(config.PROMPTS_PATH).read_text())
    _, tasks = data.load_examples()
    test_keys = set(splits["test"])  # the only place the test split is used
    test_tasks = [t for t in tasks if t["key"] in test_keys]
    instructions = {
        "one_liner": prompts.ONE_LINER,
        "baseline": saved["baseline"],
        "gepa": saved["gepa"],
    }

    results = Path(config.RESULTS_DIR)
    results.mkdir(exist_ok=True)
    meta = {
        "draws_per_task": config.TEST_DRAWS,
        "temperature": config.TEMPERATURE,
        "max_new_tokens": config.MAX_NEW_TOKENS,
        "workers": config.TEST_WORKERS,
        "instructions": instructions,
        "server": {
            "gpu": config.GPU_TYPE,
            "dtype": config.DTYPE,
            "model": config.MODEL_NAME,
            **config.SERVER_VERSIONS,
        },
    }
    meta_path = results / "test_draws_meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise ValueError(
            "test_draws_meta.json is from a run with different settings; use a new results folder"
        )
    meta_path.write_text(json.dumps(meta, indent=2))

    path = (
        results / "test_draws.jsonl"
    )  # one line per finished (instruction, task); a crash resumes from here
    finished = set()
    if path.exists():
        finished = {
            (r["prompt"], r["key"])
            for r in map(json.loads, path.read_text().splitlines())
        }
    work = [
        (name, t)
        for name in instructions
        for t in test_tasks
        if (name, t["key"]) not in finished
    ]
    calls = len(work) * config.TEST_DRAWS
    # the pilot averaged 1.54 s per call at 8 workers; 11 workers is assumed to scale to about 1.12 s
    estimate.estimate(
        [
            {
                "model": config.SERVED_NAME,
                "calls": calls,
                "seconds": calls * 1.12,
                "basis": "assumed",
            }
        ]
    )

    optimize.make_student()  # configures DSPy for the server; every instruction below runs through it
    predicts = {
        name: dspy.Predict(dspy.Signature(prompts.SIGNATURE, text))
        for name, text in instructions.items()
    }
    lock = threading.Lock()

    def run_cell(item):
        name, task = item
        record = {
            "prompt": name,
            "key": task["key"],
            "instruction_id": task["instruction_id"],
            "draws": [draw(predicts[name], task) for _ in range(config.TEST_DRAWS)],
        }
        with lock, path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    with ThreadPoolExecutor(config.TEST_WORKERS) as pool:
        list(tqdm(pool.map(run_cell, work), total=len(work)))

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for name in instructions:
        counts = Counter(
            d["outcome"] for r in rows if r["prompt"] == name for d in r["draws"]
        )
        print(name, dict(counts))
    print(
        f"finished {len(rows)} of {len(instructions) * len(test_tasks)} (instruction, task) cells"
    )


def average_logprob(token_logprobs):
    """Mean log-probability of the generated tokens (closer to 0 means more confident); None when none were generated."""
    return sum(token_logprobs) / len(token_logprobs) if token_logprobs else None


def greedy(client, instruction, task):
    """One temperature-0 answer, asked with the same messages DSPy builds. Returns the outcome, the reply and the model's confidence."""
    adapter = dspy.ChatAdapter(use_json_adapter_fallback=False)
    signature = dspy.Signature(prompts.SIGNATURE, instruction)
    messages = adapter.format(signature, [], {"prompt": task["prompt"]})
    try:
        completion = client.chat.completions.create(
            model=config.SERVED_NAME,
            messages=messages,
            temperature=0,
            max_tokens=config.MAX_NEW_TOKENS,
            logprobs=True,
        )
    except (
        openai.OpenAIError
    ) as error:  # the server or network failed; the model is not to blame
        return {
            "outcome": "request_failure",
            "reply": str(error),
            "avg_logprob": None,
            "tokens": 0,
            "finish": None,
        }
    choice = completion.choices[0]
    logprobs = [t.logprob for t in (choice.logprobs.content or [])]
    try:
        reply = adapter.parse(signature, choice.message.content)["response"]
        outcome = "pass" if checks.passes(task, reply) else "fail"
    except dspy.utils.exceptions.AdapterParseError:
        reply, outcome = choice.message.content, "parse_error"
    return {
        "outcome": outcome,
        "reply": reply,
        "avg_logprob": average_logprob(logprobs),
        "tokens": len(logprobs),
        "finish": choice.finish_reason,
    }


def run_greedy():
    """One temperature-0 answer per test task and instruction, plus a rerun of 10 tasks to see whether the server repeats itself."""
    splits = json.loads(Path(config.SPLITS_PATH).read_text())
    saved = json.loads(Path(config.PROMPTS_PATH).read_text())
    _, tasks = data.load_examples()
    test_keys = set(splits["test"])
    test_tasks = [t for t in tasks if t["key"] in test_keys]
    instructions = {
        "one_liner": prompts.ONE_LINER,
        "baseline": saved["baseline"],
        "gepa": saved["gepa"],
    }
    client = openai.OpenAI(
        base_url=os.environ["MODAL_URL"] + "/v1",
        api_key="unused",
        default_headers={
            "Modal-Key": os.environ["MODAL_ID"],
            "Modal-Secret": os.environ["MODAL_SECRET"],
        },
    )
    first = [(name, t) for name in instructions for t in test_tasks]
    again = [(name, t) for name in instructions for t in test_tasks[:10]]
    estimate.estimate(
        [
            {
                "model": config.SERVED_NAME,
                "calls": len(first) + len(again),
                "seconds": (len(first) + len(again)) * 1.12,
                "basis": "assumed",
            }
        ]
    )

    def answer(item):
        name, task = item
        return {
            "prompt": name,
            "key": task["key"],
            "instruction_id": task["instruction_id"],
            **greedy(client, instructions[name], task),
        }

    with ThreadPoolExecutor(config.TEST_WORKERS) as pool:
        results = list(tqdm(pool.map(answer, first), total=len(first)))
        reruns = list(tqdm(pool.map(answer, again), total=len(again)))
    by_cell = {(r["prompt"], r["key"]): r for r in results}
    different = sum(
        by_cell[r["prompt"], r["key"]]["reply"] != r["reply"] for r in reruns
    )

    # one live DSPy call, to check that the messages built by hand are exactly the ones DSPy sends
    lm, predict = optimize.make_student(instructions["one_liner"])
    try:
        predict(prompt=test_tasks[0]["prompt"])
    except dspy.utils.exceptions.AdapterParseError:
        pass  # the call still happened and is in the history
    adapter = dspy.ChatAdapter(use_json_adapter_fallback=False)
    built = adapter.format(
        dspy.Signature(prompts.SIGNATURE, instructions["one_liner"]),
        [],
        {"prompt": test_tasks[0]["prompt"]},
    )
    messages_match = lm.history[-1]["messages"] == built

    record = {
        "run": {
            "temperature": 0,
            "max_new_tokens": config.MAX_NEW_TOKENS,
            "server": {
                "gpu": config.GPU_TYPE,
                "dtype": config.DTYPE,
                "model": config.MODEL_NAME,
                **config.SERVER_VERSIONS,
            },
            "rerun_checked": len(reruns),
            "rerun_different_replies": different,
            "messages_match_dspy": messages_match,
        },
        "results": [
            {
                **r,
                "avg_logprob": None
                if r["avg_logprob"] is None
                else round(r["avg_logprob"], 2),
            }
            for r in results
        ],
    }
    path = Path(config.RESULTS_DIR) / "greedy.json"
    path.parent.mkdir(exist_ok=True)
    with path.open("x") as f:  # "x": never overwrite an earlier run
        json.dump(record, f, indent=2)
    counts = Counter(r["outcome"] for r in results)
    print(
        f"{len(results)} greedy answers {dict(counts)}; rerun of {len(reruns)}: {different} replies differed; messages match DSPy: {messages_match}"
    )
    print(f"wrote {path}")


def cell_rates(rows):
    """Acceptance rate per instruction and task, {instruction: {task key: passes / graded draws}}; request failures are left out."""
    rates = {}
    for r in rows:
        graded = [d for d in r["draws"] if d["outcome"] != "request_failure"]
        if graded:
            rates.setdefault(r["prompt"], {})[r["key"]] = sum(
                d["outcome"] == "pass" for d in graded
            ) / len(graded)
    return rates


def paired_differences(rates, first, second):
    """Keys of the tasks both instructions have a rate for, and second minus first for each; paired by task key, never by position."""
    keys = sorted(rates[first].keys() & rates[second].keys())
    return keys, [rates[second][k] - rates[first][k] for k in keys]


def bootstrap_interval(differences, resamples, seed):
    """95% interval for the mean difference: resample the tasks with replacement, average each resample, take the 2.5th and 97.5th percentiles."""
    diffs = np.asarray(differences)
    picks = np.random.default_rng(seed).integers(
        0, len(diffs), size=(resamples, len(diffs))
    )
    low, high = np.percentile(diffs[picks].mean(axis=1), [2.5, 97.5])
    return float(low), float(high)


def scipy_interval(first_rates, second_rates, resamples, seed):
    """The same interval from SciPy's paired bootstrap, as a check on the by-hand one."""
    result = scipy.stats.bootstrap(
        (first_rates, second_rates),
        lambda a, b, axis: np.mean(b - a, axis=axis),
        paired=True,
        vectorized=True,
        n_resamples=resamples,
        confidence_level=0.95,
        method="percentile",
        random_state=seed,
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def screenshot_task(rates):
    """The task where GEPA beat the one-liner by the most; ties go to the lower one-liner rate, then the lower key."""
    shared = rates["gepa"].keys() & rates["one_liner"].keys()
    # gains are rounded before comparing: 0.7 - 0.3 is 0.39999999999999997 in floating point, which would break a true tie with 0.4
    return min(
        shared,
        key=lambda k: (
            -round(rates["gepa"][k] - rates["one_liner"][k], 6),
            rates["one_liner"][k],
            k,
        ),
    )


def analyze():
    """Compare the three instructions on the test tasks and save the numbers to results/summary.json."""
    results = Path(config.RESULTS_DIR)
    rows = [
        json.loads(line)
        for line in (results / "test_draws.jsonl").read_text().splitlines()
    ]
    greedy_rows = json.loads((results / "greedy.json").read_text())["results"]
    rates = cell_rates(rows)
    names = ["one_liner", "baseline", "gepa"]
    type_of = {r["key"]: r["instruction_id"] for r in rows}

    comparisons = []
    for first in ("baseline", "one_liner"):
        keys, diffs = paired_differences(rates, first, "gepa")
        by_hand = bootstrap_interval(diffs, config.BOOTSTRAP_RESAMPLES, config.SEED)
        check = scipy_interval(
            [rates[first][k] for k in keys],
            [rates["gepa"][k] for k in keys],
            config.BOOTSTRAP_RESAMPLES,
            config.SEED,
        )
        comparisons.append(
            {
                "from": first,
                "to": "gepa",
                "tasks": len(keys),
                "mean_difference": round(float(np.mean(diffs)), 2),
                "by_hand_interval": [round(x, 2) for x in by_hand],
                "scipy_interval": [round(x, 2) for x in check],
            }
        )

    per_type = {}
    for kind in sorted(set(type_of.values())):
        keys = [k for k, t in type_of.items() if t == kind]
        per_type[kind] = {"tasks": len(keys)} | {
            n: round(float(np.mean([rates[n][k] for k in keys if k in rates[n]])), 2)
            for n in names
        }

    points = [
        {
            "prompt": g["prompt"],
            "key": g["key"],
            "avg_logprob": g["avg_logprob"],
            "rate": round(rates[g["prompt"]][g["key"]], 2),
        }
        for g in greedy_rows
        if g["avg_logprob"] is not None and g["key"] in rates.get(g["prompt"], {})
    ]
    spearman = {}
    for label, subset in [("all", points)] + [
        (n, [p for p in points if p["prompt"] == n]) for n in names
    ]:
        rho = scipy.stats.spearmanr(
            [p["avg_logprob"] for p in subset], [p["rate"] for p in subset]
        ).statistic
        spearman[label] = {"rho": round(float(rho), 2), "points": len(subset)}

    chosen = screenshot_task(rates)
    summary = {
        "bootstrap": {
            "resamples": config.BOOTSTRAP_RESAMPLES,
            "confidence": 0.95,
            "seed": config.SEED,
        },
        "overall": {
            n: round(float(np.mean(list(rates[n].values()))), 2) for n in names
        },
        "comparisons": comparisons,
        "per_type": per_type,
        "spearman_logprob_vs_rate": spearman,
        "screenshot_task": {
            "key": chosen,
            "instruction_id": type_of[chosen],
            "one_liner_rate": round(rates["one_liner"][chosen], 2),
            "gepa_rate": round(rates["gepa"][chosen], 2),
        },
        "points": points,
    }
    with (results / "summary.json").open(
        "x"
    ) as f:  # "x": never overwrite an earlier analysis
        json.dump(summary, f, indent=2)

    print(f"{len(type_of)} test tasks; overall pass rate {summary['overall']}")
    for c in comparisons:
        print(
            f"{c['to']} minus {c['from']}: {c['mean_difference']:+.2f}, by hand {c['by_hand_interval']}, SciPy {c['scipy_interval']} ({c['tasks']} tasks)"
        )
    print(f"\n{'type':48}{'tasks':>6}{'one_liner':>11}{'baseline':>10}{'gepa':>7}")
    for kind, row in per_type.items():
        print(
            f"{kind:48}{row['tasks']:>6}{row['one_liner']:>11.2f}{row['baseline']:>10.2f}{row['gepa']:>7.2f}"
        )
    print(f"\nSpearman, average logprob vs acceptance rate: {spearman}")

    _, all_tasks = data.load_examples()
    prompt = next(t["prompt"] for t in all_tasks if t["key"] == chosen)
    print(
        f"\nlargest GEPA gain over the one-liner: task {chosen} ({type_of[chosen]}), {summary['screenshot_task']['one_liner_rate']} -> {summary['screenshot_task']['gepa_rate']}"
    )
    print(f"prompt: {prompt[:300]}")
    for g in greedy_rows:
        if g["key"] == chosen and g["prompt"] in ("one_liner", "gepa"):
            print(f"\n[{g['prompt']}] greedy {g['outcome']}:\n{g['reply'][:500]}")


def main(mode):
    if mode == "summary":
        return analyze()
    if mode == "test-draws":
        return run_test_draws()
    if mode == "greedy":
        return run_greedy()
    splits = json.loads(Path(config.SPLITS_PATH).read_text())
    _, tasks = data.load_examples()
    tasks = pilot_tasks(tasks, splits)
    if mode == "smoke":
        tasks = tasks[:3]
    samples = config.PILOT_SAMPLES
    calls = len(tasks) * samples
    # about 6 s per call came from the M3 demo; the server runs WORKERS calls at once
    estimate.estimate(
        [
            {
                "model": config.SERVED_NAME,
                "calls": calls,
                "seconds": calls * 6 / config.WORKERS,
                "basis": "assumed",
            }
        ]
    )
    lm, predict = optimize.make_student()

    start = time.time()
    with ThreadPoolExecutor(config.WORKERS) as pool:
        run = pool.map(lambda task: acceptance_rate(predict, task, samples), tasks)
        results = list(tqdm(run, total=len(tasks)))
    seconds = time.time() - start
    output_tokens = sum(h["usage"].get("completion_tokens", 0) for h in lm.history)

    totals = {o: sum(r[o] for r in results) for o in OUTCOMES}
    print(f"\n{len(tasks)} tasks x {samples} samples: {totals}")
    print(
        f"actual: {len(lm.history)} calls, {seconds:.0f} s (about ${seconds * config.T4_PER_SECOND:.2f}), {output_tokens} output tokens"
    )

    at_top = sum(r["rate"] == 1.0 for r in results) / len(
        tasks
    )  # tasks that passed every sample
    at_bottom = sum(r["rate"] == 0.0 for r in results) / len(
        tasks
    )  # tasks that passed none; parse errors count as misses
    graded = totals["pass"] + totals["fail"] + totals["parse_error"]
    print(
        f"{at_top:.0%} of tasks passed every sample, {at_bottom:.0%} passed none, overall pass rate {totals['pass'] / graded:.0%}"
    )
    verdict = (
        "STOP, reconsider before M5" if at_top > config.STOP_ABOVE else "continue to M5"
    )
    print(
        f"stop rule (stop if more than {config.STOP_ABOVE:.0%} of tasks pass every sample): {verdict}"
    )

    if mode == "smoke":
        for task, r in zip(tasks, results):
            print(f"\n{task['instruction_id']} {task['kwargs']}\n  -> {r}")
        print("\nthe raw replies, to read by hand:")
        for h in lm.history:
            reply = str(h["outputs"][0])
            marker = "marker ok " if "[[ ## response ## ]]" in reply else "NO MARKER "
            print(marker, "|", reply[:170].replace("\n", " "))
        return

    record = {
        "run": {
            "instruction": prompts.ONE_LINER,
            "samples_per_task": samples,
            "temperature": config.TEMPERATURE,
            "max_new_tokens": config.MAX_NEW_TOKENS,
            "server": {
                "gpu": config.GPU_TYPE,
                "dtype": config.DTYPE,
                "model": config.MODEL_NAME,
                **config.SERVER_VERSIONS,
            },
            "wall_seconds": round(seconds, 2),
            "output_tokens": output_tokens,
            "totals": totals,
        },
        "tasks": [
            {
                "key": t["key"],
                "instruction_id": t["instruction_id"],
                "split": t["split"],
                **r,
                "rate": None if r["rate"] is None else round(r["rate"], 2),
            }
            for t, r in zip(tasks, results)
        ],
    }
    path = Path(config.RESULTS_DIR) / "pilot_oneliner.json"
    path.parent.mkdir(exist_ok=True)
    with path.open("x") as f:  # "x": never overwrite an earlier run
        json.dump(record, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1])
