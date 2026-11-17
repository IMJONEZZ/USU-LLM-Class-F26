"""Tests for the retrieval augmented answering.

Same arrangement as everywhere else, where the client, the encoder and the
HTTP session are all arguments and the tests pass fakes. No database, no model
download, no network.

The two that matter most are the ones guarding the rules. The extra credit says
the base model and no prompt engineering, and both of those are things I could
break later without noticing: pointing at last assignment's fine tuned endpoint
would still return sensible answers, and quietly growing the prompt template
would still run. Neither would fail anything unless something checks.
"""

import pytest

from src.rag import (
    BASE_MODEL,
    BASE_URL,
    PROMPT,
    TOP_K,
    answer_question,
    build_prompt,
    format_context,
    retrieve,
)
from src.vectordb import QA_COLLECTION


class FakeEncoder:
    def encode(self, texts, normalize_embeddings=False):
        return [[1.0, 0.0] for _ in texts]


class FakeHit:
    def __init__(self, payload, score=0.9):
        self.payload = payload
        self.score = score


class FakeResult:
    def __init__(self, points):
        self.points = points


class FakeClient:
    def __init__(self, hits=None):
        self.hits = hits or []
        self.queries = []

    def query_points(self, collection_name, query, limit, with_payload):
        self.queries.append({"name": collection_name, "limit": limit})
        return FakeResult(self.hits[:limit])


class FakeResponse:
    def __init__(self, text):
        self.status_code = 200
        self._text = text

    def json(self):
        return {"choices": [{"message": {"content": self._text}}]}

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self, text="an answer"):
        self.text = text
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return FakeResponse(self.text)


def hit(question, answer):
    return FakeHit({"text": question, "question": question, "answer": answer})


# --- building the context -----------------------------------------------------


def test_format_context_pairs_the_question_with_its_answer():
    out = format_context([{"question": "How many Ls?", "answer": "Three."}])
    assert out == "Q: How many Ls?\nA: Three."


def test_format_context_falls_back_to_the_answer_alone():
    assert format_context([{"answer": "Three."}]) == "Three."


def test_format_context_falls_back_to_raw_text():
    # The Star Wars collection has no answer field, so a document from there
    # still has to render as something rather than an empty string.
    assert format_context([{"text": "BEN: the Force"}]) == "BEN: the Force"


def test_format_context_of_nothing_is_empty():
    assert format_context([]) == ""


def test_format_context_separates_multiple_documents():
    out = format_context(
        [{"question": "a", "answer": "1"}, {"question": "b", "answer": "2"}]
    )
    assert out == "Q: a\nA: 1\n\nQ: b\nA: 2"


# --- the prompt, which is not allowed to get clever ---------------------------


def test_build_prompt_contains_the_context_and_the_question():
    prompt = build_prompt(
        "How many Ls?", [{"question": "How many Ls?", "answer": "Three."}]
    )
    assert "Three." in prompt
    assert prompt.rstrip().endswith("Answer:")
    assert "How many Ls?" in prompt


def test_the_prompt_template_stays_plain():
    """The extra credit bans prompt engineering, so this pins the template.

    If somebody later adds "you are a careful assistant, think step by step"
    the answers would probably improve and the result would stop meaning
    anything. This fails if the template grows instructions.
    """
    assert PROMPT.count("{") == 2
    assert PROMPT.strip().startswith("Context:")
    assert len(PROMPT.splitlines()) <= 5
    for word in ("you are", "step by step", "carefully", "expert", "must"):
        assert word not in PROMPT.lower()


# --- retrieval ----------------------------------------------------------------


def test_retrieve_looks_in_the_question_collection():
    client = FakeClient(hits=[hit("q", "a")])
    retrieve("q", client, FakeEncoder())
    assert client.queries[0]["name"] == QA_COLLECTION


def test_retrieve_asks_for_one_document_by_default():
    """Ten documents and a near exact match, so more context is just noise."""
    client = FakeClient(hits=[hit("q", "a")])
    retrieve("q", client, FakeEncoder())
    assert client.queries[0]["limit"] == TOP_K == 1


# --- answering ----------------------------------------------------------------


def test_answer_question_returns_what_the_model_said():
    client = FakeClient(hits=[hit("How many Ls?", "Three.")])
    out = answer_question(
        "How many Ls?", client, FakeEncoder(), session=FakeSession("Three."), key="k"
    )
    assert out["answer"] == "Three."


def test_answer_question_uses_the_base_model_not_my_fine_tune():
    """This is the rule the extra credit actually turns on.

    Assignment 7's endpoint would answer these happily and the output would
    look fine, so nothing would give it away.
    """
    session = FakeSession()
    client = FakeClient(hits=[hit("q", "a")])
    answer_question("q", client, FakeEncoder(), session=session, key="k")
    assert session.calls[0]["json"]["model"] == BASE_MODEL == "llama-3.2-1b-base"
    assert "hw8-serve-base" in session.calls[0]["url"]
    assert "hw8-serve-base" in BASE_URL


def test_answer_question_sends_the_retrieved_answer_as_context():
    session = FakeSession()
    client = FakeClient(hits=[hit("How many Ls?", "There are 3.")])
    answer_question("How many Ls?", client, FakeEncoder(), session=session, key="k")
    sent = session.calls[0]["json"]["messages"][0]["content"]
    assert "There are 3." in sent


def test_answer_question_reports_which_document_it_got():
    client = FakeClient(hits=[hit("How many Ls?", "Three.")])
    out = answer_question(
        "How many Ls?", client, FakeEncoder(), session=FakeSession(), key="k"
    )
    assert out["retrieved"] == ["How many Ls?"]
    assert out["retrieval_correct"] is True


def test_retrieval_correct_is_false_when_the_wrong_document_comes_back():
    """A wrong answer has two causes and they need telling apart.

    Either retrieval handed over the wrong document or the model had the right
    one and ignored it, and from the outside those look the same.
    """
    client = FakeClient(hits=[hit("Was Talos alive?", "Not biologically.")])
    out = answer_question(
        "How many Ls are in parallel?",
        client,
        FakeEncoder(),
        session=FakeSession(),
        key="k",
    )
    assert out["retrieval_correct"] is False


def test_retrieval_correct_is_false_when_nothing_came_back():
    client = FakeClient(hits=[])
    out = answer_question("q", client, FakeEncoder(), session=FakeSession(), key="k")
    assert out["retrieval_correct"] is False
    assert out["retrieved"] == []


def test_answer_question_does_not_sample():
    session = FakeSession()
    answer_question(
        "q", FakeClient(hits=[hit("q", "a")]), FakeEncoder(), session=session, key="k"
    )
    assert session.calls[0]["json"]["temperature"] == 0.0


@pytest.mark.parametrize("budget", [50, 400])
def test_answer_question_token_budget_is_adjustable(budget):
    session = FakeSession()
    answer_question(
        "q",
        FakeClient(hits=[hit("q", "a")]),
        FakeEncoder(),
        session=session,
        key="k",
        max_tokens=budget,
    )
    assert session.calls[0]["json"]["max_tokens"] == budget
