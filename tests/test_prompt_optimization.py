"""Tests for src/prompt_optimization.py.

CI cannot run the real model, so most tests use dspy's DummyLM, which answers with fixed
text but goes through the real DSPy program, evaluator and BootstrapFewShot optimizer.
One real-model test is skipped unless RUN_REAL_MODEL=1 (needs `ollama serve`).
"""

import json
import os

import dspy
import pytest
from dspy.utils import DummyLM

from src import prompt_optimization as po


class FixedAnswer(dspy.Module):
    """A fake program that always answers with the same text. Not an LM call."""

    def __init__(self, answer: str):
        super().__init__()
        self.answer = answer

    def forward(self, question: str) -> dspy.Prediction:
        return dspy.Prediction(answer=self.answer)


class TruthTeller(dspy.Module):
    """A fake program that always counts correctly, read from the question text."""

    def forward(self, question: str) -> dspy.Prediction:
        letter = question.split("'")[1]
        word = question.split("'")[3]
        return dspy.Prediction(answer=f"There are {word.count(letter)}")


def twos(n: int) -> list[dspy.Example]:
    """Training examples whose true answer is 2, so a model that says 2 is always right."""
    return [
        dspy.Example(question=f"q{i}", answer="2").with_inputs("question")
        for i in range(n)
    ]


# ---------- parse_count: the fair answer checker ----------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("3", "3"),
        ("There are 2 Ls in parallel.", "2"),
        ("The letter appears twice.", "2"),
        ("It appears once.", "1"),
        ("It appears three times.", "3"),
        ("none, zero times", "0"),
        ("It appears 1 time, not 2 times", "2"),  # the last number is the answer
        ("I cannot tell", None),
        ("", None),
    ],
)
def test_parse_count(text, expected):
    assert po.parse_count(text) == expected


def test_parse_count_does_not_match_words_inside_other_words():
    # "someone" contains "one" and "nobody" contains "no": neither is a count
    assert po.parse_count("someone or nobody") is None


# ---------- questions and data split ----------


def test_make_question_wording():
    assert (
        po.make_question("parallel", "l")
        == "How many times does the letter 'l' appear in the word 'parallel'?"
    )


def test_make_examples_answers_are_true_counts():
    for ex in po.make_examples(po.TEST_WORDS, 30):
        letter, word = ex.question.split("'")[1], ex.question.split("'")[3]
        assert ex.answer == str(word.count(letter))
        assert int(ex.answer) >= 1  # only letters that really occur


def test_make_examples_is_repeatable_and_has_no_duplicates():
    first = po.make_examples(po.TRAIN_WORDS, 40, seed=1)
    second = po.make_examples(po.TRAIN_WORDS, 40, seed=1)
    assert [e.question for e in first] == [e.question for e in second]
    assert len({e.question for e in first}) == 40


def test_make_examples_only_uses_given_words():
    for ex in po.make_examples(["banana"], 3):
        assert "'banana'" in ex.question


def test_make_examples_rejects_too_many_questions():
    with pytest.raises(ValueError, match="only 3 distinct"):
        po.make_examples(["banana"], 4)  # banana has 3 distinct letters


def test_make_examples_marks_question_as_input():
    ex = po.make_examples(["banana"], 1)[0]
    assert list(ex.inputs().keys()) == ["question"]


def test_train_and_test_words_never_overlap():
    assert not set(po.TRAIN_WORDS) & set(po.TEST_WORDS)
    assert len(po.TRAIN_WORDS) == 20
    assert len(po.TEST_WORDS) == 10


def test_default_sizes_are_possible_with_the_word_lists():
    assert len(po.make_examples(po.TRAIN_WORDS, po.TRAIN_SIZE)) == po.TRAIN_SIZE
    assert len(po.make_examples(po.TEST_WORDS, po.TEST_SIZE)) == po.TEST_SIZE


# ---------- metric ----------


def test_metric_right_and_wrong():
    ex = dspy.Example(question="q", answer="2").with_inputs("question")
    assert po.metric(ex, dspy.Prediction(answer="twice")) is True
    assert po.metric(ex, dspy.Prediction(answer="3")) is False
    assert po.metric(ex, dspy.Prediction(answer="no idea")) is False


# ---------- program and evaluation ----------


def test_baseline_program_is_plain_question_to_answer():
    program = po.build_program()
    assert isinstance(program, dspy.Predict)
    assert list(program.signature.input_fields) == ["question"]
    assert list(program.signature.output_fields) == ["answer"]
    assert program.demos == []  # no examples, no hints


def test_evaluate_perfect_program_scores_100():
    result = po.evaluate(TruthTeller(), po.make_examples(po.TEST_WORDS, 10))
    assert result["accuracy"] == 100.0
    assert (result["correct"], result["total"]) == (10, 10)


def test_evaluate_wrong_program_scores_0_and_records_answers():
    result = po.evaluate(FixedAnswer("99"), po.make_examples(po.TEST_WORDS, 5))
    assert result["accuracy"] == 0.0
    assert all(
        item["model_answer"] == "99" and not item["correct"] for item in result["items"]
    )


def test_evaluate_partial_score():
    # every answer is "2", so only questions whose true answer is 2 count as correct
    examples = po.make_examples(["banana"], 3)  # b=1, a=3, n=2
    result = po.evaluate(FixedAnswer("2"), examples)
    assert (result["correct"], result["total"]) == (1, 3)
    assert result["accuracy"] == pytest.approx(33.33, abs=0.01)


# ---------- optimizer and prompt text ----------


def test_optimize_adds_the_models_own_right_answers_as_examples():
    dspy.configure(lm=DummyLM([{"answer": "2"}] * 50))
    optimized = po.optimize(po.build_program(), twos(8), max_demos=3)
    demos = optimized.predictors()[0].demos
    assert 1 <= len(demos) <= 3
    assert all(po.parse_count(d["answer"]) == "2" for d in demos)


def test_optimize_keeps_only_right_answers():
    # the model says 7 but every true answer is 2, so nothing is worth keeping
    dspy.configure(lm=DummyLM([{"answer": "7"}] * 50))
    optimized = po.optimize(po.build_program(), twos(5), max_demos=3)
    assert optimized.predictors()[0].demos == []


def test_optimize_does_not_change_the_baseline_program():
    dspy.configure(lm=DummyLM([{"answer": "2"}] * 50))
    baseline = po.build_program()
    po.optimize(baseline, twos(5))
    assert baseline.demos == []


def test_render_prompt_before_has_question_and_no_examples():
    text = po.render_prompt(po.build_program(), "How many?")
    assert "How many?" in text
    assert "[system]" in text and "[user]" in text
    assert (
        "[[ ## answer ## ]]" in text.split("[user]")[0]
    )  # field format is described once
    assert text.count("How many?") == 1


def test_render_prompt_after_contains_the_examples():
    dspy.configure(lm=DummyLM([{"answer": "2"}] * 50))
    optimized = po.optimize(po.build_program(), twos(5), max_demos=2)
    text = po.render_prompt(optimized, "How many?")
    assert "q0" in text  # a training question now appears inside the prompt
    assert len(text) > len(po.render_prompt(po.build_program(), "How many?"))


# ---------- whole experiment ----------


def test_run_experiment_reports_before_after_and_prompts():
    dspy.configure(lm=DummyLM([{"answer": "2"}] * 200))
    train = twos(6)
    test = po.make_examples(["banana"], 3)
    results = po.run_experiment(
        trainset=train, testset=test, challenge_question="q-hard"
    )
    assert results["train_size"] == 6 and results["test_size"] == 3
    assert results["before"]["total"] == results["after"]["total"] == 3
    assert results["gain_points"] == pytest.approx(
        results["after"]["accuracy"] - results["before"]["accuracy"], abs=0.01
    )
    assert results["challenge_question"] == "q-hard"
    assert "q-hard" in results["prompt_before"]
    assert len(results["prompt_after"]) > len(results["prompt_before"])
    assert results["demos_in_prompt"] >= 1


def test_run_experiment_default_data_uses_the_real_split(monkeypatch):
    seen = {}

    def fake_evaluate(program, devset):
        seen.setdefault("tests", devset)
        return {"accuracy": 0.0, "correct": 0, "total": len(devset), "items": []}

    monkeypatch.setattr(po, "evaluate", fake_evaluate)
    dspy.configure(lm=DummyLM([{"answer": "2"}] * 300))
    results = po.run_experiment()
    assert (
        results["train_size"] == po.TRAIN_SIZE and results["test_size"] == po.TEST_SIZE
    )
    assert results["challenge_question"] == po.make_question("parallel", "l")
    test_words = {q.question.split("'")[3] for q in seen["tests"]}
    assert test_words <= set(po.TEST_WORDS)  # nothing from the training words


def test_save_results_writes_json(tmp_path):
    out = tmp_path / "sub" / "r.json"
    po.save_results({"a": 1}, out)
    assert json.loads(out.read_text()) == {"a": 1}


# ---------- GEPA pieces ----------


def test_make_examples_exclude_keeps_sets_apart():
    first = po.make_examples(po.TRAIN_WORDS, 40)
    second = po.make_examples(po.TRAIN_WORDS, 20, seed=1, exclude=first)
    assert not {e.question for e in first} & {e.question for e in second}


def test_reference_scores_for_lazy_answers():
    examples = po.make_examples(["banana"], 3)  # answers 1, 3, 2 in some order
    assert po.reference_scores(examples) == {"always_1": 33.33, "always_twice": 33.33}
    assert po.reference_scores([]) == {}


def test_feedback_metric_correct_answer():
    ex = dspy.Example(question=po.make_question("parallel", "l"), answer="3")
    result = po.feedback_metric(ex, dspy.Prediction(answer="three"))
    assert result.score == 1.0
    assert "Correct" in result.feedback


def test_feedback_metric_wrong_answer_spells_the_word_out():
    ex = dspy.Example(question=po.make_question("parallel", "l"), answer="3")
    result = po.feedback_metric(ex, dspy.Prediction(answer="twice"))
    assert result.score == 0.0
    assert "p-a-r-a-l-l-e-l" in result.feedback
    assert "position(s) 5, 6, 8" in result.feedback
    assert "correct count is 3" in result.feedback


def test_reflection_lm_is_creative_and_uncached():
    lm = po.reflection_lm()
    assert lm.kwargs["temperature"] == 1.0
    assert lm.cache is False


def test_gepa_optimizer_passes_budget_and_both_datasets(monkeypatch):
    seen = {}

    class FakeGEPA:
        def __init__(self, **kwargs):
            seen["init"] = kwargs

        def compile(self, program, trainset, valset):
            seen["data"] = (trainset, valset)
            return program

    monkeypatch.setattr(dspy, "GEPA", FakeGEPA)
    program, train, val = po.build_program(), twos(3), twos(2)
    result = po.make_gepa_optimizer(val, budget=77)(program, train)
    assert result is program
    assert seen["init"]["max_metric_calls"] == 77
    assert seen["init"]["metric"] is po.feedback_metric
    assert seen["data"] == (train, val)


def test_run_experiment_accepts_any_optimizer_and_records_instructions():
    dspy.configure(lm=DummyLM([{"answer": "2"}] * 50))
    calls = []

    def keep_program(program, trainset):
        calls.append(len(trainset))
        return program

    results = po.run_experiment(
        trainset=twos(4), testset=twos(3), optimizer=keep_program
    )
    assert calls == [4]
    assert results["gain_points"] == 0.0
    assert results["instructions_before"] == results["instructions_after"]
    assert results["reference_scores"]["always_twice"] == 100.0


# ---------- model setup and command line ----------


def test_configure_lm_uses_ollama_and_no_cache(monkeypatch):
    configured = {}
    monkeypatch.setattr(dspy, "configure", lambda **kw: configured.update(kw))
    lm = po.configure_lm()
    assert lm.model == po.MODEL
    assert lm.kwargs["temperature"] == 0.0
    assert lm.cache is False
    assert configured["lm"] is lm


def test_main_rejects_unknown_command(capsys):
    assert po.main([]) == 2
    assert po.main(["other"]) == 2
    assert "usage" in capsys.readouterr().out


def test_main_run_gepa_uses_separate_validation_questions(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(po, "configure_lm", lambda: None)
    seen = {}

    def fake_make_optimizer(valset):
        seen["val"] = valset
        return "optimizer"

    def fake_run(trainset, optimizer):
        seen["train"] = trainset
        seen["optimizer"] = optimizer
        return {"before": {"accuracy": 20.0}, "after": {"accuracy": 60.0}}

    monkeypatch.setattr(po, "make_gepa_optimizer", fake_make_optimizer)
    monkeypatch.setattr(po, "run_experiment", fake_run)
    assert po.main(["run-gepa"]) == 0
    assert seen["optimizer"] == "optimizer"
    assert len(seen["train"]) == po.TRAIN_SIZE and len(seen["val"]) == po.VAL_SIZE
    assert not {e.question for e in seen["train"]} & {e.question for e in seen["val"]}
    assert (tmp_path / "assignment_10/results/gepa_results.json").exists()
    assert "before: 20.0%  after: 60.0%" in capsys.readouterr().out


def test_main_run_saves_and_prints(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(po, "configure_lm", lambda: None)
    monkeypatch.setattr(
        po,
        "run_experiment",
        lambda: {"before": {"accuracy": 20.0}, "after": {"accuracy": 50.0}},
    )
    assert po.main(["run"]) == 0
    assert "before: 20.0%  after: 50.0%" in capsys.readouterr().out
    assert (tmp_path / "assignment_10/results/optimization_results.json").exists()


# ---------- real model, run by hand only ----------


@pytest.mark.skipif(
    os.environ.get("RUN_REAL_MODEL") != "1", reason="needs ollama serve and llama3.2"
)
def test_real_model_answers_letter_questions():
    po.configure_lm()
    result = po.evaluate(po.build_program(), po.make_examples(po.TEST_WORDS, 5))
    assert result["total"] == 5
    assert all(item["model_answer"] for item in result["items"])
