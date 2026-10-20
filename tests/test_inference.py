from types import SimpleNamespace
from unittest.mock import Mock

from src import inference


def test_run_inference_returns_response_text(monkeypatch):
    create = Mock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(text="The Emperor is in charge.")]
        )
    )
    client = SimpleNamespace(completions=SimpleNamespace(create=create))
    openai_client = Mock(return_value=client)
    monkeypatch.setattr(inference, "OpenAI", openai_client)

    result = inference.run_inference(
        "Who is in charge?", "https://example.modal.run/v1"
    )

    assert result == "The Emperor is in charge."
    openai_client.assert_called_once_with(
        base_url="https://example.modal.run/v1", api_key="EMPTY"
    )
    create.assert_called_once_with(
        model=inference.DEFAULT_MODEL,
        prompt="Who is in charge?",
        max_tokens=128,
    )


def test_run_inference_returns_empty_string_when_response_text_is_none(monkeypatch):
    create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(text=None)]))
    client = SimpleNamespace(completions=SimpleNamespace(create=create))
    monkeypatch.setattr(inference, "OpenAI", Mock(return_value=client))

    result = inference.run_inference("A prompt", "https://example.modal.run/v1")

    assert result == ""
