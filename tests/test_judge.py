import pytest

from src import config, judge, teacher

ROW = {
    "id": "e1",
    "category": "lateral_thinking",
    "question": "How much wood can the shelf hold if it holds 5 pounds?",
    "answer": "5 pounds",
    "accepted_alternatives": ["five pounds", "at most 5 pounds"],
    "chain": "Restate: SECRET CHAIN TEXT",
}


def _fake_judge(monkeypatch, *replies):
    prompts = []
    queue = list(replies)

    def fake(prompt, model, temperature):
        prompts.append(prompt)
        return queue.pop(0)

    monkeypatch.setattr(teacher, "_generate", fake)
    return prompts


def test_prompt_has_the_question_the_key_the_alternatives_and_the_response():
    prompt = judge.judge_prompt(ROW, "5 pounds of wood")
    for part in (ROW["question"], "5 pounds", "five pounds", "5 pounds of wood"):
        assert part in prompt


def test_prompt_leaves_out_the_teacher_chain():
    assert "SECRET CHAIN TEXT" not in judge.judge_prompt(ROW, "5 pounds of wood")


@pytest.mark.parametrize(
    ("reply", "expected"),
    [('{"correct": true}', True), ('{"correct": false}', False)],
)
def test_judge_answer_returns_the_judges_yes_or_no(monkeypatch, reply, expected):
    _fake_judge(monkeypatch, reply)
    assert judge.judge_answer(ROW, "5 pounds of wood") is expected


def test_judge_answer_uses_the_judge_model_and_temperature(monkeypatch):
    seen = []

    def fake(prompt, model, temperature):
        seen.append((model, temperature))
        return '{"correct": true}'

    monkeypatch.setattr(teacher, "_generate", fake)
    judge.judge_answer(ROW, "x")
    assert seen == [(config.JUDGE_MODEL, config.JUDGE_TEMPERATURE)]


def test_judge_answer_with_a_reply_that_is_not_yes_or_no_raises(monkeypatch):
    _fake_judge(monkeypatch, "probably")
    with pytest.raises(ValueError):
        judge.judge_answer(ROW, "x")


def test_judge_all_gives_one_verdict_per_row_in_order(monkeypatch):
    prompts = _fake_judge(monkeypatch, '{"correct": true}', '{"correct": false}')
    rows = [ROW, {**ROW, "id": "e2"}]
    graded = [{"final_answer": "5 pounds of wood"}, {"final_answer": "5,000 pounds"}]
    assert judge.judge_all(rows, graded) == [True, False]
    assert len(prompts) == 2


def test_judged_metrics_per_category_with_undefined_for_an_empty_one():
    graded = [
        {"category": "false_premise"},
        {"category": "false_premise"},
        {"category": "misconception"},
    ]
    metrics = judge.judged_metrics(graded, [True, False, True])
    assert metrics["n"] == 3
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["per_category"]["false_premise"] == {
        "n": 2,
        "correct": 1,
        "accuracy": 0.5,
    }
    assert metrics["per_category"]["time_sensitive"] == {
        "n": 0,
        "correct": 0,
        "accuracy": "undefined",
    }


def test_the_json_reply_may_be_wrapped_in_code_fences(monkeypatch):
    _fake_judge(monkeypatch, '```json\n{"correct": true}\n```')
    assert judge.judge_answer(ROW, "x") is True
