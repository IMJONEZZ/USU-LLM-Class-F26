import pytest

from src import config, gemini, rag


def test_build_prompt_is_the_document_then_the_question_and_nothing_else():
    assert rag.build_prompt("DOC", "Why?") == "DOC\n\nQuestion: Why?\nAnswer:"


def test_complete_sends_temperature_zero_stop_and_modal_headers(monkeypatch):
    monkeypatch.setenv("MODAL_URL", "https://x.test")
    monkeypatch.setenv("MODAL_ID", "id")
    monkeypatch.setenv("MODAL_SECRET", "secret")
    seen = {}

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"text": " 3 \n"}]}

    def fake_post(url, headers, json):
        seen.update(url=url, headers=headers, json=json)
        return Response()

    monkeypatch.setattr(rag.requests, "post", fake_post)
    text, retry_wait = rag.complete("P")
    assert text == "3"
    assert retry_wait == pytest.approx(0.0, abs=0.1)  # no 503s, so no waiting
    assert seen["url"] == "https://x.test/v1/completions"
    assert seen["headers"] == {"Modal-Key": "id", "Modal-Secret": "secret"}
    assert seen["json"]["temperature"] == 0
    assert seen["json"]["stop"] == config.ANSWER_STOP
    assert seen["json"]["max_tokens"] == config.ANSWER_MAX_TOKENS


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"correct": true}', True),
        ('{"correct": false}', False),
        ('```json\n{"correct":  TRUE}\n```', True),
        ('Sure! {"correct": true} hope that helps', True),
        ("", False),
        ("probably", False),
    ],
)
def test_parse_verdict_handles_messy_output_and_never_guesses_true(text, expected):
    assert rag.parse_verdict(text) is expected


def test_judge_answer_uses_the_judge_model_and_temperature(monkeypatch):
    seen = []

    def fake(prompt, model, temperature):
        seen.append((model, temperature, prompt))
        return '{"correct": true}'

    monkeypatch.setattr(gemini, "_generate", fake)
    assert rag.judge_answer("Q?", "A", "resp") is True
    model, temperature, prompt = seen[0]
    assert (model, temperature) == (config.JUDGE_MODEL, config.JUDGE_TEMPERATURE)
    assert all(part in prompt for part in ("Q?", "A", "resp"))


def test_score_counts_correct_answers():
    assert rag.score([{"answer_correct": True}, {"answer_correct": False}] * 2) == 2


def test_run_all_keeps_retrieval_and_answer_correctness_separate(monkeypatch):
    questions = [
        {"id": "a", "question": "Qa?", "answer": "1"},
        {"id": "b", "question": "Qb?", "answer": "2"},
    ]
    monkeypatch.setattr(rag.vectordb, "top1", lambda store, q: ("a", "DOC", 0.1))
    monkeypatch.setattr(rag, "complete", lambda prompt: ("1", 0.0))
    monkeypatch.setattr(rag, "judge_answer", lambda q, a, r: r == a)
    results = rag.run_all(questions, store=None)
    assert [r["retrieval_correct"] for r in results] == [True, False]
    assert [r["answer_correct"] for r in results] == [True, False]
    assert results[1]["retrieved_id"] == "a"


class _Reply:
    def __init__(self, status_code):
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise rag.requests.HTTPError(str(self.status_code))

    def json(self):
        return {"choices": [{"text": "ok"}]}


def _env(monkeypatch):
    for key in ("MODAL_URL", "MODAL_ID", "MODAL_SECRET"):
        monkeypatch.setenv(key, "x")


def test_complete_retries_a_503_then_succeeds(monkeypatch):
    _env(monkeypatch)
    replies = iter([_Reply(503), _Reply(503), _Reply(200)])
    sleeps = []
    monkeypatch.setattr(rag.requests, "post", lambda *a, **k: next(replies))
    monkeypatch.setattr(rag.time, "sleep", sleeps.append)
    clock = iter([0, 0, 15, 30])
    monkeypatch.setattr(rag.time, "monotonic", lambda: next(clock))
    assert rag.complete("P") == ("ok", 30)
    assert sleeps == [config.RETRY_INTERVAL_SECONDS] * 2


def test_complete_gives_up_on_repeated_503_after_the_timeout(monkeypatch):
    _env(monkeypatch)
    clock = iter([0, 0, 1, config.RETRY_TIMEOUT_SECONDS + 1])
    monkeypatch.setattr(rag.requests, "post", lambda *a, **k: _Reply(503))
    monkeypatch.setattr(rag.time, "sleep", lambda s: None)
    monkeypatch.setattr(rag.time, "monotonic", lambda: next(clock))
    with pytest.raises(rag.requests.HTTPError):
        rag.complete("P")


def test_complete_does_not_retry_other_errors(monkeypatch):
    _env(monkeypatch)
    calls = []
    monkeypatch.setattr(
        rag.requests, "post", lambda *a, **k: calls.append(1) or _Reply(400)
    )
    with pytest.raises(rag.requests.HTTPError):
        rag.complete("P")
    assert len(calls) == 1


def test_run_all_times_retrieval_and_completion_but_not_503_waits_or_the_judge(
    monkeypatch,
):
    questions = [{"id": "a", "question": "Qa?", "answer": "1"}]
    monkeypatch.setattr(rag.vectordb, "top1", lambda store, q: ("a", "DOC", 0.1))
    monkeypatch.setattr(rag, "complete", lambda prompt: ("1", 1.0))
    monkeypatch.setattr(rag, "judge_answer", lambda q, a, r: True)
    clock = iter([10.0, 13.5])  # a judge call would need a third reading
    monkeypatch.setattr(rag.time, "perf_counter", lambda: next(clock))
    assert rag.run_all(questions, store=None)[0]["seconds"] == 2.5


def test_mean_seconds_averages_the_rows():
    assert rag.mean_seconds([{"seconds": 1.0}, {"seconds": 4.0}]) == 2.5
