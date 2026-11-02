"""Tests for the HTML service.

Same arrangement as the last few assignments. The HTTP session is an argument
and the tests hand over a fake, so none of this needs a GPU, a model download
or a network.

Most of these are aimed at the validator, because that's the part with real
logic in it. The grammar is what actually stops bad markup from being
generated, but the grammar runs inside llama.cpp on a machine CI has never
heard of, so the thing I can pin down here is whether I'd notice if it
stopped working.
"""

from src.html_output import (
    ALLOWED_TAGS,
    BASE_URL,
    COLD_START_ATTEMPTS,
    COLD_START_SECONDS,
    MODEL,
    generate,
    grammar_tags,
    is_valid,
    show,
    validate,
)

PAGE = """<!DOCTYPE html>
<html>
<head>
<title>Tatooine</title>
</head>
<body>
<h1>Tatooine</h1>
<p>A desert world in the Outer Rim.</p>
<ul>
<li>It has two suns</li>
<li>Mostly sand</li>
</ul>
</body>
</html>"""


def test_a_page_the_grammar_would_produce_is_valid():
    assert validate(PAGE) == []
    assert is_valid(PAGE)


def test_the_grammar_and_the_validator_allow_the_same_tags():
    # If I add a tag to gpu/html.gbnf and forget to add it here, the server
    # starts emitting markup this module calls invalid, and the failure shows
    # up as a confusing runtime error instead of a failing test.
    assert grammar_tags() == set(ALLOWED_TAGS)


def test_an_unclosed_tag_is_caught():
    problems = validate("<html><body><p>hello</body></html>")
    assert any("never closed" in p for p in problems)
    assert not is_valid("<html><body><p>hello</body></html>")


def test_a_tag_outside_the_grammar_is_caught():
    assert validate("<html><body><script>alert(1)</script></body></html>") == [
        "<script> is not a tag the grammar allows"
    ]


def test_an_end_tag_with_nothing_open_is_caught():
    assert validate("</p>") == ["</p> closes something that was never opened"]


def test_attributes_are_caught():
    # The grammar has no attribute rule at all, so markup with one didn't come
    # from this server. It's also the shape an injection would take.
    problems = validate('<html><body><p onclick="x()">hi</p></body></html>')
    assert "<p> carries attributes and the grammar has none" in problems


def test_a_self_closing_tag_is_caught():
    problems = validate("<html><body><br/></body></html>")
    assert any("self closing" in p for p in problems)


def test_tags_closed_in_the_wrong_order_are_caught():
    problems = validate("<html><body><h1>hi</h2></body></html>")
    assert "</h2> closes <h1>" in problems


def test_empty_output_is_not_valid_html():
    assert validate("") == ["no HTML tags at all, this is plain text"]
    assert not is_valid("")


def test_prose_with_no_tags_is_not_valid_html():
    # This is the one I actually care about. If the server ever comes up
    # without the grammar loaded it answers in English, and every other check
    # in here passes an English sentence without complaint.
    assert not is_valid("Tatooine is a desert planet with two suns.")


# --- talking to the service ---------------------------------------------------


class FakeResponse:
    def __init__(self, text):
        self.status_code = 200
        self._text = text

    def json(self):
        return {"choices": [{"message": {"content": self._text}}]}

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self, text=PAGE):
        self.text = text
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return FakeResponse(self.text)


def test_generate_returns_the_markup_and_its_problems():
    session = FakeSession()
    result = generate("a page about Tatooine", session=session, key="k")
    assert result["topic"] == "a page about Tatooine"
    assert result["html"] == PAGE
    assert result["problems"] == []


def test_generate_sends_no_grammar_of_its_own():
    # The whole design is that the grammar belongs to the server. If someone
    # later "helpfully" starts sending one from here, there are suddenly two
    # places to keep it right, and the one in the repo is the one that'll rot.
    session = FakeSession()
    generate("anything", session=session, key="k")
    body = session.calls[0]["json"]
    assert "grammar" not in body
    assert "response_format" not in body


def test_generate_asks_the_html_service_and_not_an_older_one():
    # Spelled out rather than compared against the constants. Writing
    # `== MODEL` here looks like a test and isn't one, because changing the
    # constant changes both sides of the assertion and it passes anyway. I
    # only noticed because swapping the model for assignment 6's fine tune
    # left the whole suite green.
    session = FakeSession()
    generate("anything", session=session, key="k")
    call = session.calls[0]
    assert call["url"] == (
        "https://froostysnoowman--hw9-serve-html-serve.us-east.modal.run"
        "/v1/chat/completions"
    )
    assert call["json"]["model"] == "qwen2.5-7b-html"


def test_the_constants_are_what_generate_actually_sends():
    # The pair to the test above: that one pins the literals, this one proves
    # the constants are the thing being sent, so the two can't drift apart.
    session = FakeSession()
    generate("anything", session=session, key="k")
    call = session.calls[0]
    assert call["url"] == f"{BASE_URL}/v1/chat/completions"
    assert call["json"]["model"] == MODEL


def test_generate_lets_me_point_it_somewhere_else():
    session = FakeSession()
    generate("anything", session=session, key="k", url="http://x", model="other")
    call = session.calls[0]
    assert call["url"] == "http://x/v1/chat/completions"
    assert call["json"]["model"] == "other"


def test_generate_waits_longer_than_the_small_model_client_did():
    # A cold GPU container takes over a minute to load 4.7GB. Assignment 7's
    # 30 second budget would have called that a dead server.
    from src import inference

    assert COLD_START_ATTEMPTS * COLD_START_SECONDS > 120
    assert COLD_START_ATTEMPTS * COLD_START_SECONDS > (
        inference.MAX_ATTEMPTS * inference.RETRY_SECONDS
    )


class LoadingThenReady:
    """503s a few times like a cold container, then answers."""

    def __init__(self, loading=3, text=PAGE):
        self.loading = loading
        self.text = text
        self.slept = []

    def post(self, url, headers=None, json=None, timeout=None):
        if self.loading > 0:
            self.loading -= 1
            return LoadingResponse()
        return FakeResponse(self.text)


class LoadingResponse:
    status_code = 503

    def json(self):
        return {"error": {"message": "Loading model"}}

    def raise_for_status(self):
        return None


def test_generate_rides_out_a_cold_start():
    session = LoadingThenReady(loading=3)
    slept = []
    result = generate("anything", session=session, key="k", sleep=slept.append)
    assert result["problems"] == []
    assert len(slept) == 3


def test_generate_reports_bad_markup_instead_of_hiding_it():
    # A page that fails the check is the most interesting output this thing
    # has, so it comes back rather than turning into an exception.
    session = FakeSession(text="Sorry, I cannot help with that.")
    result = generate("anything", session=session, key="k")
    assert result["problems"] == ["no HTML tags at all, this is plain text"]


def test_show_prints_the_page_without_printing_the_key(capsys):
    session = FakeSession()
    show("a page about Tatooine", session=session, key="hunter2")
    printed = capsys.readouterr().out
    assert "REQUEST" in printed and "RESPONSE" in printed
    assert "<h1>Tatooine</h1>" in printed
    assert "hunter2" not in printed


def test_show_lists_the_problems_when_there_are_any(capsys):
    session = FakeSession(text="<html><body><p>oops</body></html>")
    show("anything", session=session, key="k")
    assert "never closed" in capsys.readouterr().out
