import json
from collections import Counter

from src import config

FIELDS = ["id", "question", "answer"]


def load_questions(path):
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if len(rows) != config.QUESTION_COUNT:
        raise ValueError(
            f"expected {config.QUESTION_COUNT} questions, found {len(rows)}"
        )
    questions = [{field: row[field] for field in FIELDS} for row in rows]
    for q in questions:
        for field in FIELDS:
            if not q[field].strip():
                raise ValueError(f"{q['id']!r}: {field} is empty")
    for question_id, count in Counter(q["id"] for q in questions).items():
        if count > 1:
            raise ValueError(f"duplicate id {question_id!r}")
    return questions
