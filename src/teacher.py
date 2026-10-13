import json
import os
import re
import time
import warnings
from collections import Counter
from pathlib import Path

import httpx
from google import genai
from google.genai import errors, types

from src import config, data

_TRANSIENT_KINDS = {"rate_limit", "timeout", "server"}
_SKIPPABLE_KINDS = {"empty", "blocked", "truncated"}
_FENCE = re.compile(r"^```[A-Za-z]*\n(.*?)\n?```$", re.DOTALL)


class TeacherAPIError(RuntimeError):
    def __init__(self, kind, message="", retry_after=None):
        super().__init__(f"{kind}: {message}" if message else kind)
        self.kind = kind
        self.retry_after = retry_after


def get_client():
    key = os.environ.get(config.API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"{config.API_KEY_ENV} is not set; export it before calling the teacher"
        )
    options = types.HttpOptions(timeout=config.TEACHER_TIMEOUT_MS)
    return genai.Client(api_key=key, http_options=options)


def _retry_after(message):
    match = re.search(r"'retryDelay': '(\d+(?:\.\d+)?)s'", message)
    return float(match.group(1)) if match else None


def _client_error_kind(code, message):
    if code in (401, 403):
        return "auth"
    if code == 429:
        return "quota" if "PerDay" in message else "rate_limit"
    return "bad_request"


def _text_or_raise(response):
    feedback = response.prompt_feedback
    if feedback is not None and feedback.block_reason:
        raise TeacherAPIError("blocked", str(feedback.block_reason))
    if not response.candidates:
        raise TeacherAPIError("empty", "no candidates")
    finish = response.candidates[0].finish_reason
    if finish == types.FinishReason.MAX_TOKENS:
        raise TeacherAPIError("truncated")
    if finish not in (None, types.FinishReason.STOP):
        raise TeacherAPIError("blocked", str(finish))
    text = response.text
    if not text or not text.strip():
        raise TeacherAPIError("empty")
    return text


def _call_model(model, prompt, temperature):
    client = get_client()
    settings = types.GenerateContentConfig(
        temperature=temperature, max_output_tokens=config.TEACHER_MAX_OUTPUT_TOKENS
    )
    try:
        response = client.models.generate_content(
            model=model, contents=prompt, config=settings
        )
    except errors.ClientError as e:
        kind = _client_error_kind(e.code, str(e))
        raise TeacherAPIError(kind, str(e), _retry_after(str(e))) from e
    except errors.ServerError as e:
        raise TeacherAPIError("server", str(e)) from e
    except httpx.TimeoutException as e:
        raise TeacherAPIError("timeout", str(e)) from e
    return _text_or_raise(response)


_call_log = []


def _now():
    return time.perf_counter()


def call_log():
    return list(_call_log)


def reset_call_log():
    _call_log.clear()


def _timed_call(model, prompt, temperature):
    started = _now()
    try:
        text = _call_model(model, prompt, temperature)
    except Exception as e:
        outcome = getattr(e, "kind", type(e).__name__)
        _call_log.append(
            {"model": model, "seconds": _now() - started, "outcome": outcome}
        )
        raise
    _call_log.append({"model": model, "seconds": _now() - started, "outcome": "ok"})
    return text


def timing_summary(records):
    by_model = {}
    for record in records:
        by_model.setdefault(record["model"], []).append(record)
    summary = {}
    for model, items in by_model.items():
        seconds = [item["seconds"] for item in items]
        summary[model] = {
            "calls": len(items),
            "total_seconds": sum(seconds),
            "mean_seconds": sum(seconds) / len(seconds),
            "max_seconds": max(seconds),
            "outcomes": dict(Counter(item["outcome"] for item in items)),
        }
    return summary


def _wait_too_long(error):
    return (
        error.retry_after is not None
        and error.retry_after > config.TEACHER_MAX_WAIT_SECONDS
    )


def _generate(prompt, model, temperature):
    for _ in range(config.TEACHER_MAX_RETRIES):
        try:
            return _timed_call(model, prompt, temperature)
        except TeacherAPIError as e:
            if e.kind not in _TRANSIENT_KINDS or _wait_too_long(e):
                raise
            time.sleep(max(config.TEACHER_RETRY_SECONDS, e.retry_after or 0))
    return _timed_call(model, prompt, temperature)


def _is_short_key(key):
    return len(key) <= config.LONG_KEY_CHARS


def chain_prompt(question, key):
    if _is_short_key(key):
        ending = f"End with '{config.FINAL_ANSWER_LABEL} {key}'."
    else:
        ending = (
            f"End with '{config.FINAL_ANSWER_LABEL}' followed by a short version "
            "of the correct answer (a few words)."
        )
    return (
        f"{config.CHAIN_INSTRUCTIONS}\n\nQuestion: {question}\n"
        f"The correct answer is: {key}\n"
        f"Write the reasoning so that it arrives at this answer. {ending}"
    )


def _unwrap(text):
    text = text.strip()
    fenced = _FENCE.match(text)
    return fenced.group(1).strip() if fenced else text


def _accept(text, key, alternatives):
    chain = _unwrap(text)
    if not config.MIN_CHAIN_CHARS <= len(chain) <= config.MAX_CHAIN_CHARS:
        return None
    try:
        data.check_chain(chain)
    except ValueError:
        return None
    if _is_short_key(key):
        final = data.extract_final_answer(chain)
        if not data.is_correct(final, key, alternatives):
            return None
    return chain


def generate_chain(row):
    key, alternatives = row["answer"], row["accepted_alternatives"]
    prompt = chain_prompt(row["question"], key)
    for _ in range(config.TEACHER_ATTEMPTS):
        try:
            text = _generate(prompt, config.TEACHER_MODEL, config.TEACHER_TEMPERATURE)
        except TeacherAPIError as e:
            if e.kind not in _SKIPPABLE_KINDS:
                raise
            continue
        chain = _accept(text, key, alternatives)
        if chain is not None:
            return {**row, "chain": chain}
    return None


def _read_store(store_path, key):
    store_path = Path(store_path)
    if not store_path.exists():
        return {}
    return {record[key]: record for record in data.read_jsonl(store_path)}


def build_training_data(rows, out_path, store_path, progress=None, generate=True):
    if not rows:
        raise ValueError("no rows to build training data from")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate id in input rows")
    store = _read_store(store_path, "id")
    for row in rows:
        stored = store.get(row["id"])
        if stored and (
            stored["question"] != row["question"] or stored["answer"] != row["answer"]
        ):
            raise ValueError(
                f"row {row['id']!r} changed since its chain was generated; "
                f"nothing was spent. Delete its line from {store_path} to "
                "regenerate it on purpose"
            )
    reset_call_log()
    row_seconds = []
    generated = reused = pending = 0
    for row in rows:
        if row["id"] in store:
            reused += 1
        elif not generate:
            pending += 1
        else:
            started = _now()
            result = generate_chain(row)
            if result is None:
                result = {
                    "id": row["id"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "category": row["category"],
                    "source": row["source"],
                    "failed": True,
                    "reason": "no_valid_chain",
                }
            store[row["id"]] = result
            data.write_jsonl(store_path, list(store.values()))
            generated += 1
            row_seconds.append(_now() - started)
        if progress is not None:
            progress()
    final = [store[i] for i in ids if i in store and not store[i].get("failed")]
    failed = [
        {"id": i, "reason": store[i]["reason"]}
        for i in ids
        if i in store and store[i].get("failed")
    ]
    if not generate and not final:
        raise ValueError("no stored chains for the requested rows; nothing to write")
    data.write_jsonl(out_path, final)
    return {
        "written": len(final),
        "generated": generated,
        "reused": reused,
        "pending": pending,
        "failed": failed,
        "timing": timing_summary(call_log()),
        "row_seconds": {
            "n": len(row_seconds),
            "mean": sum(row_seconds) / len(row_seconds) if row_seconds else None,
            "max": max(row_seconds, default=None),
        },
    }


_PER_ITEM_SKIPPABLE_KINDS = _SKIPPABLE_KINDS | {"timeout", "server"}


def _judge_prompt(source_row, variant_question):
    return (
        "A test question was reworded automatically. Decide two things.\n"
        f"Original question: {source_row['question']}\n"
        f"Correct answer to the original: {source_row['answer']}\n"
        f"Reworded question: {variant_question}\n"
        "natural: true only if the reworded question reads like a fluent, ordinary "
        "English question a person could ask.\n"
        "answer_unchanged: true only if the correct answer to the reworded question "
        "is still exactly the correct answer given above.\n"
        'Reply with JSON only, like {"natural": true, "answer_unchanged": true}.'
    )


def _parse_verdict(text):
    try:
        verdict = json.loads(_unwrap(text))
    except ValueError:
        return None
    if not isinstance(verdict, dict):
        return None
    flags = (verdict.get("natural"), verdict.get("answer_unchanged"))
    if not all(isinstance(flag, bool) for flag in flags):
        return None
    return {"natural": flags[0], "answer_unchanged": flags[1]}


def judge_variant(source_row, variant_question):
    prompt = _judge_prompt(source_row, variant_question)
    try:
        text = _generate(prompt, config.JUDGE_MODEL, config.JUDGE_TEMPERATURE)
    except TeacherAPIError as e:
        if e.kind not in _PER_ITEM_SKIPPABLE_KINDS:
            raise
        return {"keep": False, "reason": "judge_error"}
    verdict = _parse_verdict(text)
    if verdict is None:
        return {"keep": False, "reason": "invalid_judge_output"}
    if not verdict["answer_unchanged"]:
        return {"keep": False, "reason": "answer_changed"}
    if not verdict["natural"]:
        return {"keep": False, "reason": "not_natural"}
    return {"keep": True, "reason": "kept"}


def _paraphrase_prompt(question, n):
    return (
        f"Reword this question {n} different ways.\n"
        f"Question: {question}\n"
        "Rules: keep the same meaning so the correct answer does not change; keep "
        "every number, unit and negation exactly as written; do not add hints or "
        "the answer; each rewording must be a fluent, natural English question.\n"
        'Reply with JSON only, like {"rewordings": ["...", "..."]}.'
    )


def _parse_rewordings(text):
    try:
        parsed = json.loads(_unwrap(text))
    except ValueError:
        return []
    items = parsed.get("rewordings") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        return []
    return [item.strip() for item in items if isinstance(item, str) and item.strip()]


def paraphrase_question(question, n):
    prompt = _paraphrase_prompt(question, n)
    try:
        text = _generate(prompt, config.TEACHER_MODEL, config.PARAPHRASE_TEMPERATURE)
    except TeacherAPIError as e:
        if e.kind not in _PER_ITEM_SKIPPABLE_KINDS:
            raise
        return []
    return _parse_rewordings(text)


def build_eval_data(
    test_rows,
    out_path,
    review_path,
    store_path,
    per_question,
    progress=None,
    paraphrase=None,
):
    paraphrase = paraphrase or paraphrase_question
    store = _read_store(store_path, "source_id")
    for row in test_rows:
        stored = store.get(row["id"])
        if stored and stored["source_question"] != row["question"]:
            raise ValueError(
                f"test question {row['id']!r} changed since its variants were "
                f"generated; nothing was spent. Delete its line from {store_path} "
                "to regenerate it on purpose"
            )
    pool = per_question * config.EVAL_OVERSAMPLE
    paraphrase_calls = judge_calls = judge_errors = 0
    for source_row in test_rows:
        question = source_row["question"]
        if len(question.split()) < config.MIN_QUESTION_WORDS:
            continue
        record = store.setdefault(
            source_row["id"],
            {
                "source_id": source_row["id"],
                "source_question": question,
                "candidates": [],
            },
        )
        candidates = record["candidates"]
        topped_up = False
        index = kept_here = 0
        while kept_here < per_question:
            if index == len(candidates):
                if topped_up:
                    break
                topped_up = True
                paraphrase_calls += 1
                known = {data.normalize(c["question"]) for c in candidates}
                fresh = [
                    v["question"]
                    for v in data.make_variants(
                        source_row, paraphrase(question, pool), pool
                    )
                    if data.normalize(v["question"]) not in known
                ]
                if not fresh:
                    break
                candidates.extend({"question": q, "verdict": None} for q in fresh)
                data.write_jsonl(store_path, list(store.values()))
            candidate = candidates[index]
            index += 1
            if candidate["verdict"] is None:
                judge_calls += 1
                verdict = judge_variant(source_row, candidate["question"])
                if verdict["reason"] == "judge_error":
                    judge_errors += 1
                    if progress is not None:
                        progress()
                    continue
                candidate["verdict"] = verdict
                data.write_jsonl(store_path, list(store.values()))
            if candidate["verdict"]["keep"]:
                kept_here += 1
            if progress is not None:
                progress()
    kept, rejected = [], []
    for source_row in test_rows:
        record = store.get(source_row["id"])
        kept_here = 0
        for index, candidate in enumerate(record["candidates"] if record else []):
            verdict = candidate["verdict"]
            if verdict is None:
                continue
            variant = {
                **data.variant_record(source_row, candidate["question"], index),
                "verdict": verdict,
            }
            if not verdict["keep"]:
                rejected.append(variant)
            elif kept_here < per_question:
                kept.append(variant)
                kept_here += 1
    kept_sources = {record["source_id"] for record in kept}
    without_variants = [r["id"] for r in test_rows if r["id"] not in kept_sources]
    all_categories = {row["category"] for row in test_rows}
    categories_without = sorted(all_categories - {r["category"] for r in kept})
    if without_variants:
        warnings.warn(
            f"no eval variants kept for test questions: {without_variants}",
            stacklevel=2,
        )
    if categories_without:
        warnings.warn(
            f"no eval variants kept for categories: {categories_without}; "
            "their per-category accuracy will be undefined",
            stacklevel=2,
        )
    data.write_jsonl(out_path, kept)
    data.write_jsonl(review_path, rejected)
    return {
        "kept": len(kept),
        "rejected": len(rejected),
        "rejected_by_reason": dict(Counter(r["verdict"]["reason"] for r in rejected)),
        "questions_without_variants": without_variants,
        "categories_without_variants": categories_without,
        "paraphrase_calls": paraphrase_calls,
        "judge_calls": judge_calls,
        "judge_errors": judge_errors,
    }
