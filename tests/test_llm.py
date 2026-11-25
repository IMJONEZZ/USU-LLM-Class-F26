from src.llm import send_request


def test_send_request_returns_text():
    text = send_request()
    print(text)
    assert len(text) > 3
