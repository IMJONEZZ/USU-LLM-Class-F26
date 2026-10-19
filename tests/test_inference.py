import pytest
import requests

from src import config, inference, modal_server
from tests.helpers import value_after

GOOD_CHAIN = (
    "Restate: A farmer has 17 sheep; all but 9 run away.\n"
    "Check: This is a wording trap.\n"
    "Solve: All but 9 means 9 remain.\n"
    "Verify: 17 - 8 = 9.\n"
    "Final answer: 9"
)


def _without(label):
    return "\n".join(
        line for line in GOOD_CHAIN.split("\n") if not line.startswith(label)
    )


def test_format_prompt_matches_the_a6_training_prompt():
    # copied from the A6 code's real output; the served model must see these exact characters
    expected = (
        "### Instruction:\nWhere did fortune cookies originate?\n\n### Response:\n"
    )
    assert inference.format_prompt("Where did fortune cookies originate?") == expected


@pytest.mark.parametrize(
    "question",
    [
        "",
        "   ",
        f"What is {config.INSTRUCTION_MARKER.strip()} this?",
        f"What is {config.RESPONSE_MARKER.strip()} this?",
    ],
)
def test_format_prompt_rejects_empty_or_marker_questions(question):
    with pytest.raises(ValueError):
        inference.format_prompt(question)


def test_check_chain_accepts_a_well_formed_reply():
    inference.check_chain(GOOD_CHAIN)


@pytest.mark.parametrize(
    "chain",
    [
        *[_without(label) for label in config.SECTION_LABELS],
        GOOD_CHAIN + "\nSolve: again",
        GOOD_CHAIN.replace("Solve:", "Verify:", 1).replace("Verify: 17", "Solve: 17"),
        GOOD_CHAIN.replace("Final answer: 9", "Final answer:"),
        GOOD_CHAIN + f"\n{config.RESPONSE_MARKER.strip()}",
    ],
    ids=[
        *[f"missing {label}" for label in config.SECTION_LABELS],
        "duplicated Solve",
        "Solve and Verify swapped",
        "empty final answer",
        "contains a template marker",
    ],
)
def test_check_chain_rejects_a_malformed_reply(chain):
    with pytest.raises(ValueError):
        inference.check_chain(chain)


def test_completion_request_names_a_served_model_and_sends_the_training_prompt():
    command = modal_server.vllm_command(fast_boot=True)
    served = {
        value_after(command, "--served-model-name"),
        value_after(command, "--lora-modules").split("=")[0],
    }
    payload = inference.completion_request("Q?")
    assert payload["model"] in served
    assert payload["prompt"] == inference.format_prompt("Q?")
    assert payload["max_tokens"] == config.EVAL_MAX_NEW_TOKENS
    assert payload["temperature"] == 0


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def sleep(self, seconds):
        self.now += seconds

    def perf_counter(self):
        return self.now


def _serve(monkeypatch, statuses):
    """Make inference see a fake clock and a server that answers with these statuses."""
    clock = _FakeClock()
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append(url)
        response = requests.Response()
        response.status_code = statuses[min(len(calls), len(statuses)) - 1]
        return response

    for name in ("MODAL_ID", "MODAL_SECRET", "MODAL_URL"):
        monkeypatch.setenv(name, "x")
    monkeypatch.setattr(inference, "time", clock)
    monkeypatch.setattr(inference.requests, "request", fake_request)
    return calls


def test_call_waits_through_503s_and_reports_how_long(monkeypatch):
    calls = _serve(monkeypatch, [503, 503, 200])
    response, waited = inference.call("GET", "/v1/models")
    assert response.status_code == 200
    assert len(calls) == 3
    assert waited == 2 * config.RETRY_INTERVAL_SECONDS


def test_call_gives_up_on_a_server_that_stays_503(monkeypatch):
    calls = _serve(monkeypatch, [503])
    with pytest.raises(requests.HTTPError):
        inference.call("GET", "/v1/models")
    assert (
        len(calls) == config.RETRY_TIMEOUT_SECONDS // config.RETRY_INTERVAL_SECONDS + 1
    )


def test_call_does_not_retry_an_auth_error(monkeypatch):
    calls = _serve(monkeypatch, [401])
    with pytest.raises(requests.HTTPError):
        inference.call("GET", "/v1/models")
    assert len(calls) == 1
