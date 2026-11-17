"""Tests for the prompt optimizer.

The real search needs a GPU, a 4.7GB model and about half an hour, so the LM
and the optimizer are both arguments and these hand over fakes. What's left to
test is the scorer and the wiring, and the scorer is where this assignment
actually went wrong, so that's where most of these live.

Two of them exist because of specific failures. GEPA came back from a full run
with the prompt unchanged, because the scorer was pass or fail and every
candidate tied on zero. And before that it scored 100% off twenty calls,
because my feedback string was handing it the list of allowed tags instead of
making it infer them. Both are easy to reintroduce and neither one looks like
a bug from the outside, they just look like results.
"""

from itertools import pairwise
from typing import ClassVar

import dspy
import pytest
from dspy.utils import DummyLM

from src import prompt_optimization as po

TOPIC = "the planet Tatooine"

CLEAN = (
    "<!DOCTYPE html>\n<html>\n<head>\n<title>Tatooine</title>\n</head>\n<body>\n"
    "<h1>Tatooine</h1>\n<p>" + ("A desert world in the Outer Rim. " * 9) + "</p>\n"
    "<ul>\n<li>Two suns</li>\n<li>Mostly sand</li>\n</ul>\n</body>\n</html>"
)
STUB = "<html><body><p>Hi</p></body></html>"
ONE_BAD_TAG = CLEAN.replace("<h1>Tatooine</h1>", '<h1 class="x">Tatooine</h1>')
MOSTLY_BAD = (
    '<html lang="en"><head><meta charset="UTF-8"><style>a{}</style></head>'
    '<body><div class="c"><img src="x"><p>hi</p></div></body></html>'
)
PROSE = "Tatooine is a desert planet with two suns."


def fake_lm(page=CLEAN, n=400):
    return DummyLM([{"html": page}] * n)


# --- the scorer -------------------------------------------------------------


def test_a_clean_page_with_real_content_scores_one():
    assert po.score_page(TOPIC, CLEAN) == (1.0, [])


def test_prose_scores_zero():
    # No tags at all is the floor. If the server ever answers in English this
    # has to be worth less than markup that's merely wrong.
    score, problems = po.score_page(TOPIC, PROSE)
    assert score == 0.0
    assert problems


def test_the_scorer_is_graded_and_not_pass_or_fail():
    # The one that matters. A full GEPA run came back with the instructions
    # byte for byte unchanged, because every candidate scored exactly 0.0 and
    # the search had nothing to rank. If these ever collapse into ties again
    # the optimizer silently stops optimizing and still reports a number.
    ladder = [CLEAN, STUB, ONE_BAD_TAG, MOSTLY_BAD, PROSE]
    scores = [po.score_page(TOPIC, page)[0] for page in ladder]
    assert all(a > b for a, b in pairwise(scores)), scores


def test_valid_and_invalid_markup_never_overlap():
    # Valid markup has to beat invalid markup even when the writing is bad,
    # because "it parses" is the thing being measured.
    assert po.score_page(TOPIC, STUB)[0] >= po.VALID_SCORE
    assert po.score_page(TOPIC, ONE_BAD_TAG)[0] < po.VALID_SCORE
    assert po.score_page(TOPIC, MOSTLY_BAD)[0] < po.VALID_SCORE


def test_nearly_right_markup_beats_badly_wrong_markup():
    # One stray attribute against a page that's wrong everywhere. This is the
    # gradient GEPA climbs, and it only exists because the share of problems
    # is measured against the number of tags.
    assert po.score_page(TOPIC, ONE_BAD_TAG)[0] > po.score_page(TOPIC, MOSTLY_BAD)[0]


def test_a_valid_page_that_says_nothing_loses_points():
    score, problems = po.score_page(TOPIC, STUB)
    assert score < 1.0
    assert any("stub" in p for p in problems)


@pytest.mark.parametrize(
    "markup, expected",
    [
        ("<html><body><p>" + "x " * 200 + "</p></body></html>", "blocks of content"),
        (CLEAN.replace("<title>Tatooine</title>", "<title>Chess</title>"), "title"),
    ],
)
def test_the_content_floor_catches_each_way_of_cheating(markup, expected):
    assert any(expected in p for p in po.content_problems(TOPIC, markup))


# --- the metric GEPA reads --------------------------------------------------


def test_the_metric_reports_problems_without_handing_over_the_spec():
    # The regression this was written for: an earlier version appended the
    # full list of allowed tags to every piece of feedback, so GEPA hit 100%
    # on a twenty call budget by copying a spec instead of inferring one. The
    # feedback may name the tag it tripped over, never the whole allowed set.
    result = po.metric(dspy.Example(topic=TOPIC), dspy.Prediction(html=MOSTLY_BAD))
    assert result.score < po.VALID_SCORE
    assert "div" in result.feedback
    assert "<h2>" not in result.feedback
    assert "<li>" not in result.feedback


def test_the_metric_says_so_when_nothing_is_wrong():
    result = po.metric(dspy.Example(topic=TOPIC), dspy.Prediction(html=CLEAN))
    assert result.score == 1.0
    assert "Valid HTML" in result.feedback


def test_a_missing_html_field_is_scored_and_not_crashed_on():
    assert po.metric(dspy.Example(topic=TOPIC), dspy.Prediction()).score == 0.0
    assert po.accuracy(dspy.Example(topic=TOPIC), dspy.Prediction(html=None)) == 0.0


# --- the data ---------------------------------------------------------------


def test_the_splits_do_not_overlap():
    # There are no gold answers to leak here, but the optimizer still shapes
    # the prompt around whatever it sees, so a topic in both places would make
    # the held out number meaningless.
    train = {e.topic for e in po.examples("train")}
    dev = {e.topic for e in po.examples("dev")}
    test = {e.topic for e in po.examples("test")}
    assert train & dev == set()
    assert train & test == set()
    assert dev & test == set()
    assert len(train | dev | test) == len(po.TOPICS) == len(set(po.TOPICS))


def test_the_test_set_is_the_big_one():
    assert len(po.examples("test")) > len(po.examples("train"))


def test_examples_are_inputs_with_no_answer_attached():
    example = po.examples("train")[0]
    assert set(example.inputs().keys()) == {"topic"}


# --- talking to the model ---------------------------------------------------


def test_build_lm_points_at_the_assignment_10_server():
    # Spelled out instead of compared against the constants, because `== MODEL`
    # passes no matter what MODEL says. I wrote that non-test in assignment 9
    # and only caught it with mutation testing.
    lm = po.build_lm(key="k")
    assert lm.model == "openai/qwen2.5-7b"
    assert lm.kwargs["api_base"] == (
        "https://froostysnoowman--hw10-serve-plain-serve.us-east.modal.run/v1"
    )


def test_the_constants_are_what_build_lm_actually_uses():
    lm = po.build_lm(key="k")
    assert lm.model == po.MODEL
    assert lm.kwargs["api_base"] == f"{po.BASE_URL}/v1"


def test_build_lm_does_not_cache_so_the_before_and_after_are_real_calls():
    assert po.build_lm(key="k").cache is False


class LoadingThenReady:
    """503s like a cold container, then answers."""

    def __init__(self, loading=2):
        self.loading = loading
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(json)
        if self.loading > 0:
            self.loading -= 1
            return Loading()
        return Ready()


class Loading:
    status_code = 503

    def json(self):
        return {"error": {"message": "Loading model"}}

    def raise_for_status(self):
        return None


class Ready:
    status_code = 200

    def json(self):
        return {"choices": [{"message": {"content": "ok"}}]}

    def raise_for_status(self):
        return None


def test_wait_until_ready_rides_out_the_cold_start():
    # dspy treats llama.cpp's "Loading model" 503 as fatal, so the waiting has
    # to happen before dspy sees the server. Without this the first real run
    # dies on a cold container.
    session = LoadingThenReady(loading=2)
    slept = []
    po.wait_until_ready(key="k", session=session, sleep=slept.append)
    assert len(slept) == 2
    assert session.calls[0]["model"] == "qwen2.5-7b"


def test_wait_until_ready_asks_for_almost_no_tokens():
    session = LoadingThenReady(loading=0)
    po.wait_until_ready(key="k", session=session, sleep=lambda _: None)
    assert session.calls[0]["max_tokens"] == 1


# --- wiring -----------------------------------------------------------------


class StubOptimizer:
    """Stands in for GEPA so these test my wiring, not dspy's search."""

    seen: ClassVar[dict] = {}

    def __init__(self, metric, reflection_lm=None, **kwargs):
        StubOptimizer.seen = {
            "metric": metric,
            "reflection_lm": reflection_lm,
            **kwargs,
        }

    def compile(self, student, trainset, valset):
        StubOptimizer.seen["trainset"] = trainset
        StubOptimizer.seen["valset"] = valset
        tuned = student.deepcopy()
        po.predictor(tuned).signature = po.predictor(tuned).signature.with_instructions(
            "Return only the document."
        )
        return tuned


def test_optimize_hands_the_search_its_budget_and_its_metric():
    program = dspy.Predict(po.WritePage)
    lm = fake_lm()
    po.optimize(program, lm, ["train"], ["val"], optimizer=StubOptimizer)
    assert StubOptimizer.seen["metric"] is po.metric
    assert StubOptimizer.seen["max_metric_calls"] == po.MAX_METRIC_CALLS
    assert StubOptimizer.seen["trainset"] == ["train"]
    assert StubOptimizer.seen["valset"] == ["val"]


def test_optimize_reflects_with_the_same_model_unless_told_otherwise():
    # The honest weakness in this setup, pinned so it can't change quietly.
    program = dspy.Predict(po.WritePage)
    lm = fake_lm()
    po.optimize(program, lm, [], [], optimizer=StubOptimizer)
    assert StubOptimizer.seen["reflection_lm"] is lm


def test_measure_generates_each_page_once():
    # Two calls to dspy.Evaluate would give the same two numbers and generate
    # every page twice, which on a T4 is real money.
    lm = fake_lm()
    dspy.configure(lm=lm)
    devset = po.examples("dev")
    po.measure(dspy.Predict(po.WritePage), devset, num_threads=1)
    assert len(lm.history) == len(devset)


def test_measure_reports_validity_and_mean_score_separately():
    dspy.configure(lm=fake_lm())
    valid, mean, pages = po.measure(dspy.Predict(po.WritePage), po.examples("dev"), 1)
    assert valid == 100.0
    assert mean < 100.0  # the canned page's title doesn't match every topic
    assert len(pages) == len(po.examples("dev"))


def test_main_measures_optimizes_and_measures_again(capsys):
    baseline, tuned, (before, after) = po.main(
        lm=fake_lm(), optimizer=StubOptimizer, num_threads=1
    )
    printed = capsys.readouterr().out
    assert "PROMPT BEFORE" in printed and "PROMPT AFTER" in printed
    assert po.instructions_of(baseline) != po.instructions_of(tuned)
    assert before == after == 100.0


def test_main_never_touches_the_network_when_a_model_is_handed_to_it(monkeypatch):
    # CI has no GPU and no key. If main ever reaches for the real server when
    # an LM was injected, these tests stop being runnable on the runner.
    def fail(*args, **kwargs):
        raise AssertionError("tried to reach the server")

    monkeypatch.setattr(po, "wait_until_ready", fail)
    monkeypatch.setattr(po, "build_lm", fail)
    po.main(lm=fake_lm(), optimizer=StubOptimizer, num_threads=1)


def test_show_prints_the_page_and_the_verdict(capsys):
    dspy.configure(lm=fake_lm())
    po.show(TOPIC, dspy.Predict(po.WritePage), "AFTER")
    printed = capsys.readouterr().out
    assert "AFTER" in printed
    assert "<h1>Tatooine</h1>" in printed
    assert "CHECK" in printed


def test_predictor_finds_the_predictor_either_way():
    assert po.predictor(dspy.Predict(po.WritePage)).signature is po.WritePage
    assert po.predictor(dspy.ChainOfThought(po.WritePage)) is not None
    assert po.demos_of(dspy.Predict(po.WritePage)) == []


def test_main_waits_for_the_server_before_it_builds_a_client(monkeypatch):
    # Order matters and isn't obvious. dspy dies on the 503 llama.cpp returns
    # while loading, so the wait has to finish before there's a client to die.
    order = []
    lm = fake_lm()

    def waited(*args, **kwargs):
        order.append("wait")

    def built(*args, **kwargs):
        order.append("build")
        return lm

    monkeypatch.setattr(po, "wait_until_ready", waited)
    monkeypatch.setattr(po, "build_lm", built)
    po.main(optimizer=StubOptimizer, num_threads=1)
    assert order == ["wait", "build"]
