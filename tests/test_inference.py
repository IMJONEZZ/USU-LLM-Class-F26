"""The CLI exposes real HTTP results and fails clearly on request errors."""

import json
from unittest.mock import Mock

import pytest
import requests

from src import inference


def test_request_and_response(monkeypatch, capsys):
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps({"text": "three"}).encode()
    post = Mock(return_value=response)
    monkeypatch.setattr(requests, "post", post)
    assert (
        inference.main(
            [
                "hello",
                "--url",
                "http://localhost:9000/",
                "--max-new-tokens",
                "12",
                "--timeout",
                "20",
            ]
        )
        == 0
    )
    post.assert_called_once_with(
        "http://localhost:9000/generate",
        json={"prompt": "hello", "max_new_tokens": 12},
        timeout=20,
    )
    output = capsys.readouterr().out
    assert "POST http://localhost:9000/generate" in output
    assert '"prompt": "hello"' in output
    assert "HTTP 200" in output
    assert '"text": "three"' in output


@pytest.mark.parametrize(
    "error", [requests.ConnectionError("offline"), requests.Timeout("timeout")]
)
def test_network_failure(monkeypatch, capsys, error):
    monkeypatch.setattr(requests, "post", Mock(side_effect=error))
    assert inference.main([]) == 1
    assert "Request failed:" in capsys.readouterr().err


@pytest.mark.parametrize(
    "status,body", [(422, b'{"detail":"too long"}'), (200, b"not JSON")]
)
def test_http_and_invalid_json_errors(monkeypatch, capsys, status, body):
    response = requests.Response()
    response.status_code = status
    response._content = body
    monkeypatch.setattr(requests, "post", Mock(return_value=response))
    assert inference.main([]) == 1
    output = capsys.readouterr()
    assert f"HTTP {status}" in output.out
    assert "Request failed:" in output.err
    if status == 422:
        assert "too long" in output.err


@pytest.mark.parametrize(
    "args", [["--timeout", "0"], ["--max-new-tokens", "257"], [" "]]
)
def test_invalid_cli_arguments(monkeypatch, args):
    post = Mock()
    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(SystemExit) as exc:
        inference.main(args)
    assert exc.value.code == 2
    post.assert_not_called()
