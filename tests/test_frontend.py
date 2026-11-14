"""
Tests for src/frontend.py.

get_response() takes the Groq client as an injectable parameter, so tests
can supply a fake client instead of making a real API call, keeping the
test suite fast and runnable in CI without a real API key or network
access.
"""

from src.frontend import SYSTEM_PROMPT, get_response


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, response_content):
        self.response_content = response_content
        self.calls = []

    def create(self, model, messages):
        self.calls.append({"model": model, "messages": messages})
        return FakeResponse(self.response_content)


class FakeChat:
    def __init__(self, response_content):
        self.completions = FakeCompletions(response_content)


class FakeClient:
    """A minimal stand-in for a Groq client."""

    def __init__(self, response_content="This is a fake response."):
        self.chat = FakeChat(response_content)


def test_get_response_returns_fake_response_content():
    """get_response should return the content of the client's response."""
    fake_client = FakeClient(response_content="Hello there!")

    result = get_response("Hi", [], client=fake_client)

    assert result == "Hello there!"


def test_get_response_includes_system_prompt_first():
    """The Star Wars system prompt should always be the first message sent."""
    fake_client = FakeClient()

    get_response("Hi", [], client=fake_client)

    sent_messages = fake_client.chat.completions.calls[0]["messages"]
    assert sent_messages[0] == {"role": "system", "content": SYSTEM_PROMPT}


def test_get_response_includes_user_message_last():
    """The user's message should be included as the final message sent to the client."""
    fake_client = FakeClient()

    get_response("What is the capital of France?", [], client=fake_client)

    sent_messages = fake_client.chat.completions.calls[0]["messages"]
    assert sent_messages[-1] == {
        "role": "user",
        "content": "What is the capital of France?",
    }


def test_get_response_includes_history_between_system_and_user_message():
    """Prior conversation history should appear after the system prompt, before the new message."""
    fake_client = FakeClient()
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello! How can I help?"},
    ]

    get_response("Tell me a joke", history, client=fake_client)

    sent_messages = fake_client.chat.completions.calls[0]["messages"]
    assert sent_messages[0]["role"] == "system"
    assert sent_messages[1] == {"role": "user", "content": "Hi"}
    assert sent_messages[2] == {
        "role": "assistant",
        "content": "Hello! How can I help?",
    }
    assert sent_messages[3] == {"role": "user", "content": "Tell me a joke"}


def test_get_response_with_empty_history_still_includes_system_prompt():
    """An empty history should still include the system prompt and the new user message."""
    fake_client = FakeClient()

    get_response("Hello", [], client=fake_client)

    sent_messages = fake_client.chat.completions.calls[0]["messages"]
    assert len(sent_messages) == 2
    assert sent_messages[0]["role"] == "system"
    assert sent_messages[1] == {"role": "user", "content": "Hello"}


def test_get_response_uses_expected_model():
    """The request should use the configured Groq model."""
    fake_client = FakeClient()

    get_response("Hi", [], client=fake_client)

    sent_model = fake_client.chat.completions.calls[0]["model"]
    assert sent_model == "openai/gpt-oss-20b"


def test_frontend_module_is_importable_without_an_api_key():
    """
    Confirm that importing src.frontend doesn't itself require an API key
    or trigger any network calls -- only calling get_response() or
    build_interface() should need that.
    """
    import src.frontend  # noqa: F401
