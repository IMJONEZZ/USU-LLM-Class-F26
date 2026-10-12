import os
import re
import time
from collections import Counter
from pathlib import Path

import httpx
from google import genai
from google.genai import errors, types

from src import config, data

_TRANSIENT_KINDS = {"rate_limit", "timeout", "server"}
_SKIPPABLE_KINDS = {"empty", "blocked", "truncated"}
_FENCE = re.compile(r"^```[A-Za-z]*\n(.*?)\n?```$", re.DOTALL)
_INSTRUCTIONS = (
    "Answer the question with exactly five plain-text sections, each starting at "
    "the beginning of a line, in this order: "
    + ", ".join(config.SECTION_LABELS)
    + " The Check section says whether this is a familiar puzzle with a twist, a "
    "false premise, a pun, a wording trap, or a plain question. Do not use "
    "markdown or code fences. The final answer must be short and appear only on "
    f"the last line, after '{config.FINAL_ANSWER_LABEL}'."
)


class TeacherAPIError(RuntimeError):
    def __init__(self, kind, message=""):
        super().__init__(f"{kind}: {message}" if message else kind)
        self.kind = kind


def get_client():
    key = os.environ.get(config.API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"{config.API_KEY_ENV} is not set; export it before calling the teacher"
        )
    options = types.HttpOptions(timeout=config.TEACHER_TIMEOUT_MS)
    return genai.Client(api_key=key, http_options=options)


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
        raise TeacherAPIError(_client_error_kind(e.code, str(e)), str(e)) from e
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


def _generate(prompt, model, temperature):
    for _ in range(config.TEACHER_MAX_RETRIES):
        try:
            return _timed_call(model, prompt, temperature)
        except TeacherAPIError as e:
            if e.kind not in _TRANSIENT_KINDS:
                raise
            time.sleep(config.TEACHER_RETRY_SECONDS)
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
        f"{_INSTRUCTIONS}\n\nQuestion: {question}\n"
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


def ensure_output_free(out_path, force):
    if Path(out_path).exists() and not force:
        raise FileExistsError(
            f"{out_path} already exists; pass --force to overwrite it"
        )


def build_training_data(rows, out_path, partial_path=None, force=False, progress=None):
    if not rows:
        raise ValueError("no rows to build training data from")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate id in input rows")
    out_path = Path(out_path)
    ensure_output_free(out_path, force)
    if partial_path is None:
        partial_path = out_path.with_name(out_path.stem + ".partial.jsonl")
    partial_path = Path(partial_path)
    done = {}
    if partial_path.exists():
        done = {
            r["id"]: r for r in data.read_jsonl(partial_path) if r["id"] in set(ids)
        }
    resumed = len(done)
    skipped = []
    reset_call_log()
    row_seconds = []
    for row in rows:
        if row["id"] not in done:
            started = _now()
            if (result := generate_chain(row)) is None:
                skipped.append({"id": row["id"], "reason": "no_valid_chain"})
            else:
                done[row["id"]] = result
                data.write_jsonl(partial_path, list(done.values()))
            row_seconds.append(_now() - started)
        if progress is not None:
            progress()
    final = [done[i] for i in ids if i in done]
    data.write_jsonl(out_path, final)
    partial_path.unlink(missing_ok=True)
    return {
        "written": len(final),
        "resumed": resumed,
        "skipped": skipped,
        "timing": timing_summary(call_log()),
        "row_seconds": {
            "n": len(row_seconds),
            "mean": sum(row_seconds) / len(row_seconds) if row_seconds else None,
            "max": max(row_seconds, default=None),
        },
    }
