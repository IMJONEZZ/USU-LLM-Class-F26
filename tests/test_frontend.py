"""Tests for the frontend's question callback without calling Modal."""

from importlib import import_module
from unittest.mock import Mock

import modal


def test_answer_question_returns_rag_answer(monkeypatch):
    remote_answer = Mock(return_value="Expected answer")

    class FakeRAG:
        def __init__(self):
            self.answer = Mock(remote=remote_answer)

    monkeypatch.setattr(modal.Cls, "from_name", Mock(return_value=FakeRAG))
    frontend = import_module("src.frontend")

    result = frontend.answer_question("Test question")

    assert result == "Expected answer"
    remote_answer.assert_called_once_with("Test question")
