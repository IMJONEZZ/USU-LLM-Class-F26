"""Tests for the front end.

None of this needs a browser. The Gradio layer is a few lines at the bottom of
`src/frontend.py` and it isn't mine to test, so what gets tested is the
function behind it: pick a service, call it, score what came back, and turn
that into something a person can read.

The services are arguments and these hand over fakes, so there's no GPU, no
key and no network in here.

Three of these exist because of bugs this assignment actually had. The switch
has to route somewhere different in each position, the verdict has to survive
being rendered as markdown, and a service that throws has to come back as text
instead of a popup.
"""

import html
import pathlib

import pytest
import requests

from src import frontend as fe
from src.inference import ServerNotReady

CLEAN = (
    "<!DOCTYPE html>\n<html>\n<head>\n<title>The Millennium Falcon</title>\n</head>\n"
    "<body>\n<h1>The Millennium Falcon</h1>\n<p>"
    + ("A light freighter with a bad reputation. " * 8)
    + "</p>\n</body>\n</html>"
)
WITH_STRONG = CLEAN.replace("<h1>", "<h1><strong>x</strong>")
TOPIC = "the Millennium Falcon"


def fakes(grammar=CLEAN, prompt=WITH_STRONG):
    return {
        fe.MODE_GRAMMAR: lambda topic, **kwargs: grammar,
        fe.MODE_PROMPT: lambda topic, **kwargs: prompt,
    }


# --- the switch -------------------------------------------------------------


def test_each_switch_position_calls_a_different_service():
    # The whole application is this. If both positions quietly went to the
    # same server every screenshot would still look right and the comparison
    # would be a lie.
    called = []
    services = {
        fe.MODE_GRAMMAR: lambda topic, **kw: called.append("grammar") or CLEAN,
        fe.MODE_PROMPT: lambda topic, **kw: called.append("prompt") or CLEAN,
    }
    fe.answer(TOPIC, fe.MODE_GRAMMAR, services)
    fe.answer(TOPIC, fe.MODE_PROMPT, services)
    assert called == ["grammar", "prompt"]


def test_the_two_modes_are_not_the_same_string():
    assert fe.MODE_GRAMMAR != fe.MODE_PROMPT
    assert set(fe.SERVICES) == set(fe.MODES)


def test_the_topic_reaches_the_service_unchanged():
    seen = []
    services = {fe.MODE_GRAMMAR: lambda topic, **kw: seen.append(topic) or CLEAN}
    fe.answer("  the Battle of Hoth  ", fe.MODE_GRAMMAR, services)
    assert seen == ["  the Battle of Hoth  "]


# --- scoring and the verdict ------------------------------------------------


def test_a_clean_page_is_valid():
    result = fe.answer(TOPIC, fe.MODE_GRAMMAR, fakes())
    assert result["problems"] == []
    assert fe.verdict(result["problems"]) == "Valid. Nothing outside the grammar."


def test_a_page_with_a_stray_tag_is_invalid_and_names_the_tag():
    result = fe.answer(TOPIC, fe.MODE_PROMPT, fakes())
    assert result["problems"]
    assert "<strong>" in fe.verdict(result["problems"])


def test_the_verdict_counts_and_truncates():
    problems = [f"problem {i}" for i in range(7)]
    line = fe.verdict(problems)
    assert line.startswith("Invalid, 7 problems.")
    assert "and 4 more" in line


def test_the_verdict_is_escaped_before_it_reaches_the_browser():
    # Gradio renders chat messages as markdown. Unescaped, "<strong> is not a
    # tag the grammar allows" renders as bold text with the tag swallowed, so
    # the one word worth reading is the one that disappears. I only caught
    # this by looking at a screenshot.
    _, _, line = fe.panels(fe.answer(TOPIC, fe.MODE_PROMPT, fakes()))
    assert "&lt;strong&gt;" in line
    assert "<strong>" not in line


def test_panels_sends_the_same_markup_to_both_sides():
    rendered, raw, _ = fe.panels(fe.answer(TOPIC, fe.MODE_GRAMMAR, fakes()))
    assert rendered == raw == CLEAN


# --- failures come back as words --------------------------------------------


def test_a_cold_server_says_so_instead_of_raising():
    def cold(topic, **kwargs):
        raise ServerNotReady("still loading after 20 attempts")

    _, _, line = fe.panels(fe.answer(TOPIC, fe.MODE_GRAMMAR, {fe.MODE_GRAMMAR: cold}))
    assert "waking up" in line
    assert "Traceback" not in line


def test_a_broken_service_comes_back_readable():
    def boom(topic, **kwargs):
        raise RuntimeError("connection reset")

    result = fe.answer(TOPIC, fe.MODE_GRAMMAR, {fe.MODE_GRAMMAR: boom})
    assert result["markup"] == ""
    assert "RuntimeError" in result["note"]
    assert "connection reset" in result["note"]


def test_a_failure_does_not_get_scored_as_invalid_html():
    # An empty page would otherwise score as "no HTML tags at all", which
    # blames the model for something the network did.
    def boom(topic, **kwargs):
        raise RuntimeError("nope")

    result = fe.answer(TOPIC, fe.MODE_GRAMMAR, {fe.MODE_GRAMMAR: boom})
    assert result["problems"] == []


# --- what the UI calls ------------------------------------------------------


def test_respond_returns_a_chat_line_and_three_panels():
    out = fe.respond(TOPIC, [], fe.MODE_GRAMMAR, fakes())
    assert len(out) == 4
    chat, rendered, raw, topic = out
    assert chat.startswith(fe.MODE_GRAMMAR)
    assert rendered == raw == CLEAN
    assert topic == TOPIC


def test_respond_hands_the_topic_back_for_the_compare_box():
    # ChatInterface clears its own textbox on submit, so the compare button
    # would have nothing to read without this.
    assert fe.respond("  the Battle of Hoth ", [], fe.MODE_GRAMMAR, fakes())[3] == (
        "the Battle of Hoth"
    )


@pytest.mark.parametrize("empty", ["", "   ", None])
def test_an_empty_message_asks_for_a_topic_instead_of_calling_the_server(empty):
    def never(topic, **kwargs):
        raise AssertionError("should not have been called")

    chat, rendered, raw, topic = fe.respond(
        empty, [], fe.MODE_GRAMMAR, {fe.MODE_GRAMMAR: never}
    )
    assert "topic" in chat.lower()
    assert rendered == raw == topic == ""


def test_compare_runs_both_services_and_returns_six_values():
    out = fe.compare(TOPIC, fakes())
    assert len(out) == 6
    assert out[2] == "Valid. Nothing outside the grammar."
    assert "&lt;strong&gt;" in out[5]


def test_compare_puts_grammar_on_the_left_and_the_prompt_on_the_right():
    # The columns are labeled in the layout and nothing checks that the values
    # arrive in the same order, so this is the check.
    out = fe.compare(TOPIC, fakes(grammar=CLEAN, prompt=WITH_STRONG))
    assert out[0] == CLEAN
    assert out[3] == WITH_STRONG


def test_compare_with_no_topic_still_fills_every_panel():
    out = fe.compare("  ", fakes())
    assert len(out) == 6
    assert all(isinstance(value, str) for value in out)


# --- the data the page is built from ----------------------------------------


def test_the_examples_carry_a_topic_and_a_real_mode():
    # Gradio wants a row per example once there are additional inputs. A row
    # with a mode that isn't in MODES would fail only when somebody clicked it.
    rows = [[topic, fe.MODE_GRAMMAR] for topic in fe.EXAMPLES]
    assert rows
    for topic, mode in rows:
        assert topic.strip()
        assert mode in fe.MODES


def test_both_example_sets_are_offered():
    assert set(fe.PLAIN_EXAMPLES) <= set(fe.EXAMPLES)
    assert set(fe.STAR_WARS_EXAMPLES) <= set(fe.EXAMPLES)
    assert len(set(fe.EXAMPLES)) == len(fe.EXAMPLES)


def test_the_optimized_instruction_is_the_one_i_measured(tmp_path):
    saved = tmp_path / "prompt.json"
    saved.write_text('{"after_instructions": "do the thing"}', encoding="utf-8")
    assert fe.optimized_instructions(saved) == "do the thing"


def test_the_committed_instruction_file_loads():
    assert fe.optimized_instructions().strip()


# --- the layout -------------------------------------------------------------


def test_the_compare_button_is_wired_to_as_many_outputs_as_compare_returns():
    # The click handler lists its outputs by hand and nothing lines them up
    # with what `compare` returns. One extra component and every panel after
    # it silently shows the wrong value.
    app = fe.build_app(fakes())
    wiring = next(
        dep
        for dep in app.get_config_file()["dependencies"]
        if dep.get("api_name") == "compare"
    )
    assert len(wiring["outputs"]) == len(fe.compare(TOPIC, fakes()))
    assert len(wiring["inputs"]) == 1


def test_the_app_builds_with_both_modes_on_the_switch():
    app = fe.build_app(fakes())
    values = [
        choice
        for block in app.blocks.values()
        if type(block).__name__ == "Radio"
        for choice, _ in block.choices
    ]
    assert fe.MODE_GRAMMAR in values
    assert fe.MODE_PROMPT in values


def test_the_page_says_what_it_is():
    assert "grammar" in fe.TITLE.lower()
    assert html.escape(fe.DESCRIPTION) == fe.DESCRIPTION


# --- the two service adapters -----------------------------------------------


def test_grammar_service_hands_back_markup_and_not_the_whole_result(monkeypatch):
    # html_output.generate returns a dict with the problems attached. The UI
    # scores the markup itself, so this adapter has to unwrap it.
    from src import html_output

    seen = {}

    def fake_generate(topic, **kwargs):
        seen.update({"topic": topic, **kwargs})
        return {"topic": topic, "html": CLEAN, "problems": ["ignored"]}

    monkeypatch.setattr(html_output, "generate", fake_generate)
    assert fe.grammar_service(TOPIC, key="k") == CLEAN
    assert seen == {"topic": TOPIC, "key": "k", "timeout": fe.COLD_SECONDS}


def test_prompt_service_uses_the_instruction_i_measured(monkeypatch):
    import dspy
    from dspy.utils import DummyLM

    import src.prompt_optimization as po

    monkeypatch.setattr(po, "wait_until_ready", lambda *a, **k: None)
    monkeypatch.setattr(po, "build_lm", lambda *a, **k: DummyLM([{"html": CLEAN}] * 4))

    markup = fe.prompt_service(TOPIC, instructions="write a page, carefully")
    assert markup == CLEAN
    assert dspy.settings.lm is not None


def test_prompt_service_falls_back_to_the_committed_instruction(monkeypatch):
    from dspy.utils import DummyLM

    import src.prompt_optimization as po

    monkeypatch.setattr(po, "wait_until_ready", lambda *a, **k: None)
    monkeypatch.setattr(po, "build_lm", lambda *a, **k: DummyLM([{"html": CLEAN}] * 4))

    used = {}
    real = fe.optimized_instructions

    def spy(path=None):
        used["called"] = True
        return real(path)

    monkeypatch.setattr(fe, "optimized_instructions", spy)
    assert fe.prompt_service(TOPIC) == CLEAN
    assert used["called"]


# --- the cold GPU ------------------------------------------------------------


def test_a_read_timeout_is_retried_because_the_first_call_wakes_the_container():
    # `ask` waits out llama.cpp's 503, but a properly asleep server holds the
    # connection instead and requests gives up at its own timeout, which never
    # reaches that retry loop. I hit this on the last check before calling the
    # assignment done, which is exactly what a grader opening a cold URL gets.
    calls = []

    def cold_then_ready(topic, **kwargs):
        calls.append(topic)
        if len(calls) == 1:
            raise requests.ReadTimeout("read timed out")
        return CLEAN

    result = fe.answer(TOPIC, fe.MODE_GRAMMAR, {fe.MODE_GRAMMAR: cold_then_ready})
    assert len(calls) == 2
    assert result["note"] is None
    assert result["markup"] == CLEAN


def test_a_server_that_never_wakes_says_so_rather_than_retrying_forever():
    calls = []

    def always_timeout(topic, **kwargs):
        calls.append(topic)
        raise requests.ConnectionError("connection refused")

    result = fe.answer(TOPIC, fe.MODE_GRAMMAR, {fe.MODE_GRAMMAR: always_timeout})
    assert len(calls) == fe.COLD_ATTEMPTS
    assert "waking up" in result["note"]


def test_a_real_error_is_not_retried():
    # Retrying a bug just makes it happen twice. Only the waking-up family
    # gets a second go.
    calls = []

    def boom(topic, **kwargs):
        calls.append(topic)
        raise RuntimeError("that is a genuine bug")

    result = fe.answer(TOPIC, fe.MODE_GRAMMAR, {fe.MODE_GRAMMAR: boom})
    assert len(calls) == 1
    assert "genuine bug" in result["note"]


def test_the_grammar_service_waits_longer_than_the_cpu_default(monkeypatch):
    # The 120 second default in src/inference.py was sized for an 800MB model
    # on CPU. A cold GPU container spends longer than that booting, and there
    # is no response to retry on while it does, so this has to be bigger.
    from src import html_output, inference

    seen = {}
    monkeypatch.setattr(
        html_output, "generate", lambda topic, **kw: seen.update(kw) or {"html": CLEAN}
    )
    fe.grammar_service(TOPIC)
    assert seen["timeout"] == fe.COLD_SECONDS
    assert fe.COLD_SECONDS > inference.TIMEOUT_SECONDS


def test_a_caller_can_still_override_the_timeout(monkeypatch):
    from src import html_output

    seen = {}
    monkeypatch.setattr(
        html_output, "generate", lambda topic, **kw: seen.update(kw) or {"html": CLEAN}
    )
    fe.grammar_service(TOPIC, timeout=5)
    assert seen["timeout"] == 5


def test_the_frontend_parses_on_the_python_the_container_runs():
    # This file runs on 3.14 locally and on whatever gpu/frontend_app.py pins
    # in the image. Python 3.14 accepts `except A, B:` without parentheses and
    # 3.12 does not, so I shipped a SyntaxError that ruff, pytest and a local
    # import all thought was fine, and only the deploy logs ever saw it.
    import ast
    import re

    app_source = pathlib.Path("gpu/frontend_app.py").read_text(encoding="utf-8")
    pinned = re.search(r'python_version="(\d+)\.(\d+)"', app_source)
    assert pinned, "frontend_app.py should pin the image's python version"
    version = (int(pinned.group(1)), int(pinned.group(2)))

    source = pathlib.Path("src/frontend.py").read_text(encoding="utf-8")
    ast.parse(source, feature_version=version)
