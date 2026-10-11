import ast
import csv
import io
import urllib.request
import warnings

from src import config, data

_FALSEQA_COMMIT = "9a9729974d8c5cfbfc6929e5b830111c0ed7c0c7"
_FALSEQA_URL = (
    "https://raw.githubusercontent.com/thunlp/FalseQA/{commit}/dataset/{split}.csv"
)
_FALSEQA_SPLITS = ("train", "valid", "test")
_FRESHQA_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1_8mi-yuK30mvoDJu1KQXD6ODem7MKMcIgVAwDSzJkjM/export?format=csv"
)

_TQA_COLUMNS = (
    "question",
    "category",
    "best_answer",
    "correct_answers",
    "incorrect_answers",
)
_TQA_CATEGORIES = {
    "Misconceptions": "misconception",
    "Misconceptions: Topical": "misconception",
    "Superstitions": "misconception",
    "Myths and Fairytales": "misconception",
    "Mandela Effect": "misconception",
    "Indexical Error: Time": "time_sensitive",
}
_BT_COLUMNS = ("id", "question", "answer", "label", "choice_list", "kind")
_BT_KINDS = {"sentence": "lateral_thinking", "word": "pun_ambiguity"}
_BT_VARIANT_SUFFIXES = ("_SR", "_CR")
_FRESH_COLUMNS = ("id", "question", "effective_year", "false_premise", "fact_type")


def _fetch_text(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8-sig")


def _load_hf(name, cfg):
    from datasets import load_dataset

    return list(load_dataset(name, cfg)["validation"])


def _require(record, columns, where):
    for column in columns:
        if column not in record:
            raise ValueError(f"{where}: missing column {column!r}")


def _clean(value, field, where):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: {field} is blank")
    return value.strip()


def _alternatives(answer, candidates, exclude=()):
    seen = {data.normalize(answer)}
    kept = []
    for candidate in candidates:
        key = data.normalize(candidate)
        if not key or key in seen or key in exclude:
            continue
        seen.add(key)
        kept.append(candidate.strip())
    return kept


def parse_truthfulqa(records):
    rows = []
    for i, record in enumerate(records):
        where = f"truthfulqa row {i}"
        _require(record, _TQA_COLUMNS, where)
        category = _TQA_CATEGORIES.get(record["category"])
        if category is None:
            continue
        question = _clean(record["question"], "question", where)
        answer = _clean(record["best_answer"], "best_answer", where)
        if not isinstance(record["correct_answers"], list):
            raise ValueError(  # noqa: TRY004 - bad data, one exception type
                f"{where}: correct_answers must be a list"
            )
        incorrect = {data.normalize(a) for a in record["incorrect_answers"]}
        rows.append(
            {
                "id": f"tqa-{i:04d}",
                "question": question,
                "answer": answer,
                "accepted_alternatives": _alternatives(
                    answer, record["correct_answers"], exclude=incorrect
                ),
                "category": category,
                "source": "truthfulqa",
            }
        )
    return rows


def _falseqa_answers(raw, where):
    raw = _clean(raw, "answer", where)
    if raw.startswith("["):
        try:
            items = ast.literal_eval(raw)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"{where}: answer is not a valid list") from e
    else:
        items = [raw]
    items = [_clean(item, "answer", where) for item in items]
    if not items:
        raise ValueError(f"{where}: answer list is empty")
    return items


def parse_falseqa(records, split):
    rows = []
    for i, record in enumerate(records):
        where = f"falseqa {split} row {i}"
        _require(record, ("question", "answer", "label"), where)
        label = record["label"]
        if label not in ("0", "1"):
            raise ValueError(f"{where}: label must be '0' or '1', got {label!r}")
        if label == "0":
            continue
        question = _clean(record["question"], "question", where)
        answer, *rest = _falseqa_answers(record["answer"], where)
        rows.append(
            {
                "id": f"falseqa-{split}-{i:04d}",
                "question": question,
                "answer": answer,
                "accepted_alternatives": _alternatives(answer, rest),
                "category": "false_premise",
                "source": "falseqa",
            }
        )
    return rows


def parse_brainteaser(records):
    rows = []
    for record in records:
        _require(record, _BT_COLUMNS, f"brainteaser {record.get('id', '?')}")
        puzzle_id = record["id"]
        where = f"brainteaser {puzzle_id}"
        if puzzle_id.endswith(_BT_VARIANT_SUFFIXES):
            continue
        category = _BT_KINDS.get(record["kind"])
        if category is None:
            raise ValueError(f"{where}: kind must be 'sentence' or 'word'")
        question = _clean(record["question"], "question", where)
        answer = _clean(record["answer"], "answer", where)
        choices, label = record["choice_list"], record["label"]
        if isinstance(label, bool) or not isinstance(label, int):
            raise ValueError(  # noqa: TRY004 - bad data, one exception type
                f"{where}: label {label!r} is not an index"
            )
        if not 0 <= label < len(choices):
            raise ValueError(f"{where}: label {label} is outside choice_list")
        normalized_choices = [data.normalize(c) for c in choices]
        if normalized_choices[label] != data.normalize(answer):
            warnings.warn(
                f"{where}: dropped, answer is not choice_list[label]", stacklevel=2
            )
            continue
        if len(set(normalized_choices)) != len(choices):
            warnings.warn(f"{where}: dropped, duplicate choices", stacklevel=2)
            continue
        if data.normalize(answer) == "none of above":
            warnings.warn(f"{where}: dropped, answer is 'None of above.'", stacklevel=2)
            continue
        rows.append(
            {
                "id": f"brainteaser-{puzzle_id}",
                "question": question,
                "answer": answer,
                "accepted_alternatives": [],
                "category": category,
                "source": "brainteaser",
            }
        )
    return rows


def parse_freshqa(text):
    table = list(csv.reader(io.StringIO(text, newline="")))
    header_at = next((i for i, r in enumerate(table) if r and r[0] == "id"), None)
    if header_at is None:
        raise ValueError("freshqa: no header row starting with 'id'")
    header = table[header_at]
    _require(dict.fromkeys(header), (*_FRESH_COLUMNS, "answer_0"), "freshqa")
    year = str(config.REFERENCE_YEAR)
    rows = []
    for cells in table[header_at + 1 :]:
        if not cells or not cells[0].strip():
            continue
        record = dict(zip(header, cells, strict=False))
        where = f"freshqa id {record['id']}"
        premise = record.get("false_premise", "").strip()
        if premise not in ("TRUE", "FALSE"):
            raise ValueError(f"{where}: false_premise must be TRUE or FALSE")
        changing = record.get("fact_type", "").strip() != "never-changing"
        if changing and record.get("effective_year", "").strip() != year:
            continue
        if premise == "TRUE":
            category = "false_premise"
        elif changing:
            category = "time_sensitive"
        else:
            continue
        question = _clean(record.get("question"), "question", where)
        answers = [record.get(f"answer_{k}", "").strip() for k in range(10)]
        answers = [a for a in answers if a]
        if not answers:
            raise ValueError(f"{where}: no answer in answer_0..answer_9")
        rows.append(
            {
                "id": f"freshqa-{record['id'].strip()}",
                "question": question,
                "answer": answers[0],
                "accepted_alternatives": _alternatives(answers[0], answers[1:]),
                "category": category,
                "source": "freshqa",
            }
        )
    return rows


def load_truthfulqa():
    return parse_truthfulqa(_load_hf("truthfulqa/truthful_qa", "generation"))


def load_falseqa():
    rows = []
    for split in _FALSEQA_SPLITS:
        url = _FALSEQA_URL.format(commit=_FALSEQA_COMMIT, split=split)
        records = list(csv.DictReader(io.StringIO(_fetch_text(url), newline="")))
        rows.extend(parse_falseqa(records, split))
    return rows


def load_brainteaser():
    return parse_brainteaser(data.read_jsonl(config.BRAINTEASER_PATH))


def load_freshqa():
    return parse_freshqa(_fetch_text(_FRESHQA_URL))


def drop_duplicate_questions(rows):
    seen_ids = set()
    seen_questions = set()
    kept = []
    dropped = 0
    for row in rows:
        if row["id"] in seen_ids:
            raise ValueError(f"duplicate id {row['id']!r}")
        seen_ids.add(row["id"])
        key = data.normalize(row["question"])
        if key in seen_questions:
            dropped += 1
            continue
        seen_questions.add(key)
        kept.append(row)
    if dropped:
        warnings.warn(f"dropped {dropped} duplicate questions", stacklevel=2)
    return kept


def load_all():
    loaders = (
        ("truthfulqa", load_truthfulqa),
        ("falseqa", load_falseqa),
        ("brainteaser", load_brainteaser),
        ("freshqa", load_freshqa),
    )
    rows = []
    for name, loader in loaders:
        loaded = loader()
        if not loaded:
            raise ValueError(f"{name}: loader returned no rows")
        rows.extend(loaded)
    for row in rows:
        data.check_schema(row)
    return drop_duplicate_questions(rows)
