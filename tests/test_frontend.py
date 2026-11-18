"""Tests for the Assignment 11 chat frontend (src/frontend.py).

CI has no Ollama, so the model is a fake: DSPy's DummyLM, or a stub program with a fixed
answer. The reply logic, the prompt switch, the scoring and the pass bar run for real. One
test also builds the Gradio app and calls it through Gradio's own test client, so a broken
connection between the buttons and the code would show up here.
"""

import json

import dspy
import pytest
from dspy.utils import DummyLM
from gradio_client import Client

from src import frontend as fe
from src.prompt_optimization import TEST_SIZE, TEST_WORDS, make_examples

GEPA_TEXT = "Count the letter carefully."


class StubProgram:
    """Stands in for a DSPy program: always gives the same answer."""

    def __init__(self, answer: str):
        self.answer = answer
        self.questions: list[str] = []

    def __call__(self, question: str) -> dspy.Prediction:
        self.questions.append(question)
        return dspy.Prediction(answer=self.answer)


@pytest.fixture
def results_file(tmp_path):
    path = tmp_path / "gepa_results.json"
    path.write_text(json.dumps({"instructions_after": GEPA_TEXT}))
    return path


def test_load_programs_has_both_prompts_and_gepa_uses_saved_instruction(results_file):
    programs = fe.load_programs(results_file)
    assert list(programs) == fe.CHOICES
    plain = programs[fe.PLAIN].predictors()[0].signature.instructions
    gepa = programs[fe.GEPA].predictors()[0].signature.instructions
    assert gepa == GEPA_TEXT
    assert plain != GEPA_TEXT


def test_reply_uses_the_program_picked_by_the_switch():
    plain, gepa = StubProgram("1"), StubProgram("2")
    reply = fe.make_reply({fe.PLAIN: plain, fe.GEPA: gepa})
    assert reply("q", [], fe.PLAIN) == "1"
    assert reply("q", [], fe.GEPA) == "2"
    assert len(plain.questions) == 1
    assert len(gepa.questions) == 1


def test_reply_strips_the_message_before_asking_the_model():
    program = StubProgram("2")
    fe.make_reply({fe.PLAIN: program})("  hello  ", [], fe.PLAIN)
    assert program.questions == ["hello"]


@pytest.mark.parametrize("message", ["", "   ", None])
def test_reply_to_an_empty_message_does_not_call_the_model(message):
    program = StubProgram("2")
    answer = fe.make_reply({fe.PLAIN: program})(message, [], fe.PLAIN)
    assert answer == fe.EMPTY_MESSAGE
    assert program.questions == []


def test_reply_to_an_unknown_prompt_choice_says_so():
    program = StubProgram("2")
    answer = fe.make_reply({fe.PLAIN: program})("q", [], "Mystery prompt")
    assert "Unknown prompt choice" in answer
    assert fe.PLAIN in answer
    assert program.questions == []


def test_reply_runs_a_real_dspy_program_with_the_fake_model(results_file):
    dspy.configure(lm=DummyLM([{"answer": "twice"}]))
    reply = fe.make_reply(fe.load_programs(results_file))
    assert (
        reply("How many times does the letter 'a' appear in 'banana'?", [], fe.PLAIN)
        == "twice"
    )


def test_score_through_reply_counts_right_answers_and_accepts_number_words():
    testset = make_examples(TEST_WORDS, 6)
    answers = {e.question: e.answer for e in testset}
    first = testset[0].question

    def reply(message, history, choice):
        if message == first:
            return "twice" if answers[message] == "2" else answers[message]
        return "99"

    scored = fe.score_through_reply(reply, testset, fe.PLAIN)
    assert scored["total"] == 6
    assert scored["correct"] == 1
    assert scored["accuracy"] == pytest.approx(16.67)
    assert [i["correct"] for i in scored["items"]] == [True] + [False] * 5


def test_score_through_reply_on_an_empty_test_set_is_zero():
    scored = fe.score_through_reply(lambda m, h, c: "1", [], fe.PLAIN)
    assert scored == {"accuracy": 0.0, "correct": 0, "total": 0, "items": []}


def scripted_reply(testset, plain_right: int, gepa_right: int):
    """A reply that gets the first N questions right for each prompt, so scores are exact."""
    truth = {e.question: e.answer for e in testset}
    order = {e.question: i for i, e in enumerate(testset)}
    limit = {fe.PLAIN: plain_right, fe.GEPA: gepa_right}

    def reply(message, history, choice):
        if order[message] < limit[choice]:
            return truth[message]
        return "99"

    return reply


def test_measure_passes_when_both_rules_hold():
    testset = make_examples(TEST_WORDS, TEST_SIZE)
    results = fe.measure(scripted_reply(testset, 12, 25), testset)  # 20% and 41.67%
    assert results["plain"]["accuracy"] == 20.0
    assert results["gepa"]["accuracy"] == pytest.approx(41.67)
    assert results["gain_points"] == pytest.approx(21.67)
    assert results["pass_bar"] == {
        "plain_within_5_of_a10": True,
        "gepa_within_5_of_a10": True,
        "gain_at_least_15": True,
    }
    assert results["passed"] is True


def test_measure_fails_when_gepa_gains_too_little():
    testset = make_examples(TEST_WORDS, TEST_SIZE)
    results = fe.measure(scripted_reply(testset, 12, 15), testset)  # 20% and 25%
    assert results["pass_bar"]["gain_at_least_15"] is False
    assert results["pass_bar"]["gepa_within_5_of_a10"] is False
    assert results["passed"] is False


def test_measure_fails_when_a_score_drifts_far_from_assignment_10():
    testset = make_examples(TEST_WORDS, TEST_SIZE)
    results = fe.measure(scripted_reply(testset, 30, 50), testset)  # 50% and 83%
    assert results["pass_bar"]["plain_within_5_of_a10"] is False
    assert results["passed"] is False


def test_measure_uses_the_assignment_10_test_questions_by_default():
    results = fe.measure(lambda m, h, c: "1")
    assert results["test_size"] == TEST_SIZE


def test_the_gradio_app_answers_through_its_own_test_client():
    plain, gepa = StubProgram("plain says 1"), StubProgram("gepa says 2")
    app = fe.build_app(fe.make_reply({fe.PLAIN: plain, fe.GEPA: gepa}))
    _, url, _ = app.launch(prevent_thread_lock=True, quiet=True)
    try:
        client = Client(url, verbose=False)
        question = "How many times does the letter 'a' appear in the word 'banana'?"
        assert client.predict(question, fe.PLAIN, api_name="/reply") == "plain says 1"
        assert client.predict(question, fe.GEPA, api_name="/reply") == "gepa says 2"
    finally:
        app.close()
    assert plain.questions == [question]
    assert gepa.questions == [question]


def test_the_radio_switch_lists_both_prompts_and_starts_on_plain():
    app = fe.build_app(lambda m, h, c: "x")
    radios = [b for b in app.blocks.values() if type(b).__name__ == "Radio"]
    assert len(radios) == 1
    assert [c[1] for c in radios[0].choices] == fe.CHOICES
    assert radios[0].value == fe.PLAIN


def test_save_results_creates_the_folder(tmp_path):
    path = tmp_path / "deep" / "out.json"
    fe.save_results({"a": 1}, path)
    assert json.loads(path.read_text()) == {"a": 1}


def test_main_without_a_valid_command_prints_usage(capsys):
    assert fe.main([]) == 2
    assert fe.main(["nonsense"]) == 2
    assert "usage" in capsys.readouterr().out


def test_main_measure_saves_results_and_prints_the_scores(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(fe, "configure_lm", lambda: None)
    monkeypatch.setattr(fe, "load_programs", dict)
    monkeypatch.setattr(fe, "make_reply", lambda programs: "reply")
    fake = {
        "plain": {"accuracy": 20.0},
        "gepa": {"accuracy": 41.67},
        "gain_points": 21.67,
        "passed": True,
    }
    monkeypatch.setattr(fe, "measure", lambda reply: fake)
    assert fe.main(["measure"]) == 0
    saved = json.loads((tmp_path / fe.FRONTEND_RESULTS).read_text())
    assert saved["passed"] is True
    out = capsys.readouterr().out
    assert "plain: 20.0%" in out
    assert "gepa: 41.67%" in out
