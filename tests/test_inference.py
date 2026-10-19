import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

INFERENCE_FILE = Path(__file__).resolve().parents[1] / "src" / "inference.py"


@pytest.fixture
def inference(monkeypatch):
    response = Mock()
    response.json.return_value = {"choices": [{"message": {"content": "A mountain"}}]}
    post = Mock(return_value=response)
    monkeypatch.setattr(requests, "post", post)

    namespace = runpy.run_path(str(INFERENCE_FILE))
    post.reset_mock()
    response.raise_for_status.reset_mock()
    return namespace, post, response


def test_sends_riddle_and_prints_answer_and_response_time(
    inference, monkeypatch, capsys
):
    namespace, post, response = inference
    times = iter((10.0, 11.25))
    monkeypatch.setitem(
        namespace["get_riddle_answer"].__globals__, "perf_counter", lambda: next(times)
    )

    namespace["get_riddle_answer"]("What has roots?")

    post.assert_called_once()
    args, kwargs = post.call_args
    assert args == ("http://127.0.0.1:8000/v1/chat/completions",)
    assert kwargs["json"]["model"] == "riddles"
    assert kwargs["json"]["messages"][-1] == {
        "role": "user",
        "content": "What has roots?",
    }
    assert kwargs["json"]["temperature"] == 0
    assert kwargs["json"]["max_tokens"] == 16
    assert kwargs["timeout"] == 120
    response.raise_for_status.assert_called_once_with()
    output = capsys.readouterr().out
    assert "Answer: A mountain" in output
    assert "Response time: 1.25 seconds" in output


def test_http_error_is_raised_without_printing_an_answer(inference, capsys):
    namespace, _, response = inference
    capsys.readouterr()
    response.raise_for_status.side_effect = requests.HTTPError("server error")

    with pytest.raises(requests.HTTPError, match="server error"):
        namespace["get_riddle_answer"]("A riddle")

    assert capsys.readouterr().out == ""


def test_script_submits_all_riddles_and_prints_expected_answers(monkeypatch, capsys):
    response = Mock()
    response.json.return_value = {"choices": [{"message": {"content": "answer"}}]}
    post = Mock(return_value=response)
    monkeypatch.setattr(requests, "post", post)

    runpy.run_path(str(INFERENCE_FILE))

    assert post.call_count == 10
    prompts = [
        call.kwargs["json"]["messages"][-1]["content"] for call in post.call_args_list
    ]
    assert len(set(prompts)) == 10
    output = capsys.readouterr().out
    for expected in (
        "A mountain",
        "Teeth",
        "Wind",
        "The sun",
        "Dark",
        "An egg",
        "Fish",
        "Fish on a table, man on a stool, cat gets the scraps",
        "Time",
    ):
        assert f"Expected answer: {expected}" in output
