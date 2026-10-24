import json

import pytest

from src import vectordb


def _rows(n=10):
    return [
        {
            "id": f"q-{i}",
            "question": f"Question {i}?",
            "answer": f"Answer {i}",
            "accepted_alternatives": ["x"],
            "category": "c",
            "chain": "Restate: long chain text",
        }
        for i in range(n)
    ]


def _write(tmp_path, rows):
    path = tmp_path / "test_set.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_load_questions_keeps_only_id_question_and_answer_in_file_order(tmp_path):
    loaded = vectordb.load_questions(_write(tmp_path, _rows()))
    assert len(loaded) == 10
    assert loaded[0] == {"id": "q-0", "question": "Question 0?", "answer": "Answer 0"}
    assert [q["id"] for q in loaded] == [f"q-{i}" for i in range(10)]


def test_load_questions_rejects_a_row_count_other_than_ten(tmp_path):
    for n in (9, 11):
        with pytest.raises(ValueError, match="10"):
            vectordb.load_questions(_write(tmp_path, _rows(n)))


def test_load_questions_rejects_duplicate_ids(tmp_path):
    rows = _rows()
    rows[3]["id"] = rows[2]["id"]
    with pytest.raises(ValueError, match="q-2"):
        vectordb.load_questions(_write(tmp_path, rows))


@pytest.mark.parametrize("field", ["id", "question", "answer"])
@pytest.mark.parametrize("empty", ["", "   "])
def test_load_questions_rejects_an_empty_field(tmp_path, field, empty):
    rows = _rows()
    rows[4][field] = empty
    with pytest.raises(ValueError, match=field):
        vectordb.load_questions(_write(tmp_path, rows))
