import json
import os
import random
import re
import tempfile
import unicodedata
from pathlib import Path

from src import config

_FINAL_LABEL_WORDS = config.FINAL_ANSWER_LABEL.rstrip(":").split()
_FINAL_LABEL_RE = re.compile(
    r"^[ \t]*" + r"[ \t]+".join(_FINAL_LABEL_WORDS) + r"[ \t]*:",
    re.IGNORECASE | re.MULTILINE,
)
_MARKERS = [config.INSTRUCTION_MARKER.strip(), config.RESPONSE_MARKER.strip()]
_APOSTROPHES = "'’‘"


def _reject_markers(text, what):
    for marker in _MARKERS:
        if marker in text:
            raise ValueError(f"{what} contains template marker {marker!r}")


def format_prompt(question):
    _reject_markers(question, "question")
    question = question.strip()
    if not question:
        raise ValueError("question is empty")
    return f"{config.INSTRUCTION_MARKER}{question}\n\n{config.RESPONSE_MARKER}"


def _check_chain(chain):
    _reject_markers(chain, "chain")
    lines = chain.split("\n")
    positions = []
    for label in config.SECTION_LABELS:
        found = [i for i, line in enumerate(lines) if line.lstrip().startswith(label)]
        if len(found) != 1:
            raise ValueError(f"expected exactly one {label!r} line, found {len(found)}")
        positions.append(found[0])
    if positions != sorted(positions):
        raise ValueError("chain sections are out of order")
    final_line = lines[positions[-1]].lstrip()
    if not final_line[len(config.FINAL_ANSWER_LABEL) :].strip():
        raise ValueError("final answer is empty")


def format_example(question, chain, eos_token):
    chain = chain.strip()
    if chain.endswith(eos_token):
        chain = chain[: -len(eos_token)].strip()
    _check_chain(chain)
    return format_prompt(question) + chain + eos_token


def format_fewshot(examples, target_question, max_chars=None):
    target_norm = normalize(target_question)
    pieces = []
    for example in examples:
        if normalize(example["question"]) == target_norm:
            raise ValueError("a few-shot example is identical to the target question")
        chain = example["chain"].strip()
        _check_chain(chain)
        pieces.append(format_prompt(example["question"]) + chain)
    pieces.append(format_prompt(target_question))
    out = "\n\n".join(pieces)
    if max_chars is not None and len(out) > max_chars:
        raise ValueError(f"few-shot prompt is {len(out)} chars, limit is {max_chars}")
    return out


def extract_final_answer(text):
    cut = len(text)
    for marker in _MARKERS:
        idx = text.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    text = text[:cut]
    match = _FINAL_LABEL_RE.search(text)
    if match is None:
        return None
    parts = []
    for line in text[match.end() :].split("\n"):
        stripped = line.strip()
        if not stripped:
            break
        if parts and (
            _FINAL_LABEL_RE.match(line)
            or any(stripped.startswith(label) for label in config.SECTION_LABELS)
        ):
            break
        parts.append(stripped)
    answer = " ".join(parts)
    return answer or None


def normalize(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).casefold().replace("−", "-")
    chars = []
    for i, ch in enumerate(text):
        if ch in _APOSTROPHES:
            continue
        is_minus = (
            ch == "-"
            and text[i + 1 : i + 2].isdigit()
            and not text[i - 1 : i].isalnum()
        )
        if unicodedata.category(ch).startswith("P") and ch != "." and not is_minus:
            chars.append(" ")
        else:
            chars.append(ch)
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", "".join(chars))
    return " ".join(text.split())


def is_correct(predicted, key, alternatives=()):
    accepted = {normalize(k) for k in (key, *alternatives)} - {""}
    if not normalize(key):
        raise ValueError("answer key is empty")
    predicted_norm = normalize(predicted)
    return predicted_norm != "" and predicted_norm in accepted


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path}: line {line_no}: invalid JSON ({e.msg})"
                ) from e
            if not isinstance(row, dict):
                raise ValueError(  # noqa: TRY004 - bad data, one exception type
                    f"{path}: line {line_no}: expected a JSON object"
                )
            rows.append(row)
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def check_schema(row):
    for key in ("id", "question", "answer", "source"):
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty string, got {value!r}")
    if row.get("category") not in config.CATEGORIES:
        raise ValueError(f"category {row.get('category')!r} is not in CATEGORIES")
    alternatives = row.get("accepted_alternatives")
    if not isinstance(alternatives, list):
        raise ValueError(  # noqa: TRY004 - bad data, one exception type
            "accepted_alternatives must be a list"
        )
    if any(not isinstance(a, str) or not a.strip() for a in alternatives):
        raise ValueError("accepted_alternatives must contain only non-empty strings")
    if "chain" in row and not isinstance(row["chain"], str):
        raise ValueError("chain must be a string when present")


def stratified_sample(rows, n, seed):
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    groups = {}
    for row in rows:
        if "category" not in row:
            raise ValueError(f"row has no 'category': {row!r}")
        groups.setdefault(row["category"], []).append(row)
    rng = random.Random(seed)
    sampled = []
    for group in groups.values():
        picked = rng.sample(range(len(group)), min(n, len(group)))
        sampled.extend(group[i] for i in sorted(picked))
    return sampled
