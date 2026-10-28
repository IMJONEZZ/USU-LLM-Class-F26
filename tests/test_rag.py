import json

import chromadb
import pytest

from src import rag, vectordb
from tests.test_vectordb import fake_embed, squad_file, word_count


@pytest.fixture
def collection(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "rag_db"))
    items = vectordb.load_squad(squad_file(tmp_path)) + rag.qa_items()
    collection, _ = vectordb.build_collection(client, items, fake_embed, "rag_test")
    return collection


class RecordingLlm:
    """Stands in for Ollama: records every prompt and answers with a fixed text."""

    def __init__(self):
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return "an answer"


def test_there_are_ten_questions_with_answers():
    assert len(rag.QUESTIONS) == 10
    assert all(question and answer for question, answer in rag.QUESTIONS)


def test_qa_items_have_ids_and_hold_question_and_answer():
    items = rag.qa_items()
    assert [item["id"] for item in items] == [f"qa{n:02d}" for n in range(1, 11)]
    question, answer = rag.QUESTIONS[8]
    assert items[8]["context"] == f"Question: {question}\nAnswer: {answer}"
    assert items[8]["question"] == question


def test_build_prompt_is_the_fixed_plain_template():
    prompt = rag.build_prompt("Who?", ["first note", "second note"])
    assert prompt == "Context:\nfirst note\n\nsecond note\n\nQuestion: Who?\nAnswer:"


def test_ask_ollama_sends_greedy_request_without_system_prompt(monkeypatch):
    sent = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"message": {"content": "hi there"}}).encode()

    def fake_urlopen(request, timeout):
        sent["url"] = request.full_url
        sent["body"] = json.loads(request.data)
        sent["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(rag.urllib.request, "urlopen", fake_urlopen)
    assert rag.ask_ollama("Question?", "some-model:1b") == "hi there"
    assert sent["url"] == rag.OLLAMA_URL
    assert sent["body"]["model"] == "some-model:1b"
    assert sent["body"]["messages"] == [{"role": "user", "content": "Question?"}]
    assert sent["body"]["stream"] is False
    assert sent["body"]["options"] == {"temperature": 0, "num_predict": 512}
    assert sent["timeout"] == 300


def test_answer_without_retrieval_sends_the_bare_question():
    llm = RecordingLlm()
    result = rag.answer_question("Was Talos alive?", llm)
    assert llm.prompts == ["Was Talos alive?"]
    assert result["retrieved_ids"] == []
    assert result["answer"] == "an answer"
    assert result["seconds"] >= 0


def test_answer_with_retrieval_puts_the_notes_in_the_prompt(collection):
    llm = RecordingLlm()
    question = rag.QUESTIONS[8][0]
    result = rag.answer_question(question, llm, collection, fake_embed, k=3)
    assert "qa09" in result["retrieved_ids"]
    assert llm.prompts[0].startswith("Context:\n")
    assert llm.prompts[0].endswith(f"Question: {question}\nAnswer:")
    assert "There are 3 Ls in the word parallel." in llm.prompts[0]


def test_retrieval_needs_an_embedding_function(collection):
    with pytest.raises(ValueError):
        rag.answer_question("Anything?", RecordingLlm(), collection)


def test_run_all_asks_every_question_both_ways_after_a_warm_up(collection):
    llm = RecordingLlm()
    results = rag.run_all(llm, collection, fake_embed, k=3)
    assert llm.prompts[0] == "Hello"
    assert len(llm.prompts) == 1 + 10 * 2
    assert [row["number"] for row in results["rows"]] == list(range(1, 11))
    assert results["own_notes_retrieved"] == 10
    assert results["median_seconds_with"] >= 0
    assert results["median_seconds_without"] >= 0


def test_run_all_counts_a_missing_own_note(collection):
    results = rag.run_all(RecordingLlm(), collection, fake_embed, k=1)
    assert 0 <= results["own_notes_retrieved"] <= 10
    assert results["k"] == 1


def test_format_results_shows_both_answers_for_each_question(collection):
    results = rag.run_all(RecordingLlm(), collection, fake_embed)
    text = rag.format_results(results)
    assert "own note in the top k: 10 of 10 questions" in text
    assert text.count("WITHOUT:") == 10
    assert text.count("WITH:") == 10


def test_main_saves_results_with_the_model_name(tmp_path, monkeypatch, capsys):
    data = squad_file(tmp_path)
    monkeypatch.setattr(
        vectordb, "default_embedder", lambda: (fake_embed, word_count, 256)
    )
    monkeypatch.setattr(vectordb, "download_squad", lambda path: data)
    monkeypatch.setattr(rag, "ask_ollama", lambda prompt, model: f"{model} says hi")
    out = tmp_path / "results" / "rag.json"
    args = [
        "--data-dir",
        str(tmp_path),
        "--db-path",
        str(tmp_path / "main_db"),
        "--n",
        "4",
        "--model",
        "test-model:1b",
        "--out",
        str(out),
    ]

    assert rag.main(args) == 0
    saved = json.loads(out.read_text())
    assert saved["model"] == "test-model:1b"
    assert len(saved["rows"]) == 10
    assert saved["rows"][0]["with_retrieval"]["answer"] == "test-model:1b says hi"
    assert "WITHOUT:" in capsys.readouterr().out
