import difflib
import hashlib
import json
import os
import random
import re
import tempfile
import unicodedata
import warnings
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


def check_chain(chain):
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
    check_chain(chain)
    return format_prompt(question) + chain + eos_token


def format_fewshot(examples, target_question, max_chars=None):
    target_norm = normalize(target_question)
    pieces = []
    for example in examples:
        if normalize(example["question"]) == target_norm:
            raise ValueError("a few-shot example is identical to the target question")
        chain = example["chain"].strip()
        check_chain(chain)
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
    sampled = []
    for category, group in groups.items():
        order = list(range(len(group)))
        random.Random(f"{seed}:{category}").shuffle(order)
        picked = sorted(order[: min(n, len(group))])
        sampled.extend(group[i] for i in picked)
    return sampled


def drop_excluded(rows, excluded_ids):
    return [row for row in rows if row["id"] not in excluded_ids]


def _check_test_row(row):
    check_schema(row)
    chain = row.get("chain")
    if not isinstance(chain, str) or not chain.strip():
        raise ValueError("chain is required")
    check_chain(chain.strip())
    final = extract_final_answer(chain)
    if not is_correct(final, row["answer"], row["accepted_alternatives"]):
        raise ValueError(
            f"the reference chain's final answer {final!r} does not match "
            f"the key {row['answer']!r}"
        )
    if (
        row["category"] == "time_sensitive"
        and row.get("reference_year") != config.REFERENCE_YEAR
    ):
        raise ValueError(
            f"time_sensitive rows need reference_year == {config.REFERENCE_YEAR}"
        )


def check_test_set(rows):
    if not rows:
        raise ValueError("no rows in the test set")
    seen = set()
    for row in rows:
        row_id = row.get("id")
        try:
            _check_test_row(row)
        except ValueError as e:
            raise ValueError(f"test row {row_id!r}: {e}") from e
        if row_id in seen:
            raise ValueError(f"duplicate test id {row_id!r}")
        seen.add(row_id)


def load_test_set(path=None, excluded=None):
    path = config.TEST_SET_PATH if path is None else path
    excluded = config.EXCLUDED_TEST_IDS if excluded is None else excluded
    rows = drop_excluded(read_jsonl(path), excluded)
    rows = [{"source": "hand_written", **row} for row in rows]
    check_test_set(rows)
    return rows


def _similarity(a, b):
    limit = config.NEAR_DUPLICATE_RATIO
    matcher = difflib.SequenceMatcher(None, a, b)
    if matcher.real_quick_ratio() < limit or matcher.quick_ratio() < limit:
        return None
    ratio = matcher.ratio()
    return ratio if ratio >= limit else None


def drop_leaked(train_rows, test_rows):
    test_questions = [normalize(row["question"]) for row in test_rows]
    exact = set(test_questions)
    kept, dropped = [], []
    for row in train_rows:
        question = normalize(row["question"])
        leaks = question in exact or any(
            _similarity(question, other) is not None for other in test_questions
        )
        if leaks:
            dropped.append(row["id"])
        else:
            kept.append(row)
    if dropped:
        warnings.warn(
            f"dropped {len(dropped)} train rows that duplicate a test question: "
            f"{dropped}",
            stacklevel=2,
        )
    return kept, dropped


def _near_duplicates(test_rows, train_rows):
    train_questions = [(r["id"], normalize(r["question"])) for r in train_rows]
    found = []
    for row in test_rows:
        question = normalize(row["question"])
        for train_id, other in train_questions:
            ratio = _similarity(question, other)
            if ratio is not None:
                found.append(
                    {"test_id": row["id"], "train_id": train_id, "ratio": ratio}
                )
                warnings.warn(
                    f"test row {row['id']!r} is {ratio:.2f} similar to "
                    f"train row {train_id!r}",
                    stacklevel=3,
                )
    return found


def check_leakage(test_rows, train_rows, eval_rows=()):
    if not train_rows:
        raise ValueError("train data is empty; nothing to check the test set against")
    test_ids = [row["id"] for row in test_rows]
    if len(set(test_ids)) != len(test_ids):
        raise ValueError("duplicate test id")
    train_questions = {}
    train_chains = {}
    for row in train_rows:
        train_questions.setdefault(normalize(row["question"]), row["id"])
        if row.get("chain"):
            train_chains.setdefault(row["chain"].strip(), row["id"])
    for row in test_rows:
        same_question = train_questions.get(normalize(row["question"]))
        if same_question is not None:
            raise ValueError(
                f"test row {row['id']!r} has the same question as "
                f"train row {same_question!r}"
            )
        same_chain = train_chains.get((row.get("chain") or "").strip())
        if same_chain is not None:
            raise ValueError(
                f"test row {row['id']!r} has the same chain as train row {same_chain!r}"
            )
    for row in eval_rows:
        in_train = train_questions.get(normalize(row["question"]))
        if in_train is not None:
            raise ValueError(f"eval row {row['id']!r} appears in train as {in_train!r}")
    return _near_duplicates(test_rows, train_rows)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_test_set_hash(test_path, hash_path):
    rows = len(read_jsonl(test_path))
    write_jsonl(hash_path, [{"sha256": file_sha256(test_path), "rows": rows}])


def check_test_set_hash(test_path, hash_path):
    current = file_sha256(test_path)
    if not Path(hash_path).exists():
        return "missing"
    (record,) = read_jsonl(hash_path)
    if record["sha256"] == current:
        return "ok"
    warnings.warn(
        f"the test set changed since its hash was recorded in {hash_path}",
        stacklevel=2,
    )
    return "changed"


_NEGATION_WORDS = frozenset(
    {
        "not",
        "no",
        "never",
        "nor",
        "none",
        "nobody",
        "nothing",
        "neither",
        "cannot",
        "without",
    }
)
_NUMBER = r"\d+(?:\.\d+)?"


def _facts(text):
    numbers = sorted(re.findall(_NUMBER, text))
    units = sorted(
        unit.lower().removesuffix("s")
        for unit in re.findall(_NUMBER + r"\s*([A-Za-z%]+)", text)
    )
    words = re.findall(r"[a-z']+", text.lower())
    negations = sum(w in _NEGATION_WORDS or w.endswith("n't") for w in words)
    return numbers, units, negations


def variant_record(row, question, index):
    return {
        "id": f"{row['id']}-v{index}",
        "source_id": row["id"],
        "question": question,
        "answer": row["answer"],
        "accepted_alternatives": list(row["accepted_alternatives"]),
        "category": row["category"],
        "chain": row["chain"],
        "source": "paraphrase",
    }


def make_variants(row, candidates, cap):
    question = row["question"]
    if len(question.split()) < config.MIN_QUESTION_WORDS:
        return []
    seen = {normalize(question)}
    variants = []
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        key = normalize(candidate)
        if key in seen or _facts(candidate) != _facts(question):
            continue
        if any(marker in candidate for marker in _MARKERS):
            continue
        seen.add(key)
        variants.append(variant_record(row, candidate, len(variants)))
        if len(variants) == cap:
            break
    return variants


def pick_random(rows, n, seed):
    picked = random.Random(seed).sample(range(len(rows)), min(n, len(rows)))
    return [rows[i] for i in sorted(picked)]
