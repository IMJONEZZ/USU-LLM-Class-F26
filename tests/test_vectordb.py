"""Exercise the exported workflow using real Chroma and offline model stand-ins."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import chromadb
import numpy as np
import pytest
import torch


@pytest.fixture
def workflow(monkeypatch, tmp_path):
    state = SimpleNamespace(prompts=[], decoded=[], database_paths=[])

    class Embedder:
        def __init__(self, *args, **kwargs):
            self.device = kwargs["device"]
            self.questions = []

        def encode(self, texts, normalize_embeddings):
            assert normalize_embeddings
            if not self.questions:
                self.questions = list(texts)
            vectors = []
            for text in texts:
                vector = np.zeros(10, dtype=np.float32)
                if text == "partially similar question":
                    vector[0], vector[1] = 0.8, 0.6
                else:
                    index = self.questions.index(text) if text in self.questions else 8
                    vector[index] = 1
                vectors.append(vector)
            return np.array(vectors)

    class Inputs(dict):
        def to(self, device):
            assert device == "cuda"
            return self

    class Tokenizer:
        eos_token_id = 0

        @classmethod
        def from_pretrained(cls, name):
            state.tokenizer_name = name
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            state.prompts.append(messages)
            return Inputs(input_ids=torch.tensor([[11, 12]]))

        def decode(self, tokens, skip_special_tokens):
            state.decoded.append(tokens.tolist())
            return "  Generated answer  "

    class Model:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            state.model_name = name
            return cls()

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            return torch.cat([kwargs["input_ids"], torch.tensor([[99, 100]])], dim=1)

    persistent_client = chromadb.PersistentClient

    def temporary_client(path, **kwargs):
        state.database_paths.append(Path(path))
        return persistent_client(path=str(tmp_path / "chroma"), **kwargs)

    write_text = Path.write_text

    def temporary_results(path, text, **kwargs):
        if path.name == "vectordb_results.json":
            state.results_path = path
            path = tmp_path / path.name
        return write_text(path, text, **kwargs)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=Embedder),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModelForCausalLM=Model, AutoTokenizer=Tokenizer),
    )
    monkeypatch.setitem(sys.modules, "marimo", SimpleNamespace(md=lambda text: None))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "Test GPU")
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(chromadb, "PersistentClient", temporary_client)
    monkeypatch.setattr(Path, "write_text", temporary_results)
    # Launch away from the repository to check that storage paths use __file__.
    script = Path(__file__).resolve().parents[1] / "src" / "vectordb.py"
    monkeypatch.chdir(tmp_path)
    state.namespace = runpy.run_path(str(script))
    state.saved_results = json.loads((tmp_path / "vectordb_results.json").read_text())
    return state


def test_question_answer_records_are_normalized(workflow):
    qa = workflow.namespace["qa"]
    assert len(qa) == 10
    for question, answer in qa.items():
        assert question and answer
        assert question == " ".join(question.split())
        assert answer == " ".join(answer.split())


def test_stores_answers_and_updates_without_duplicates(workflow):
    collection = workflow.namespace["collection"]
    record = collection.get(ids=["q0"], include=["documents", "metadatas"])
    question = next(iter(workflow.namespace["qa"]))
    answer = workflow.namespace["qa"][question]
    assert record["documents"] == [f"Question: {question}\nAnswer: {answer}"]
    assert record["metadatas"] == [{"question": question, "answer": answer}]
    collection.upsert(
        ids=["q0"],
        embeddings=workflow.namespace["embeddings"][:1].tolist(),
        documents=["Updated answer"],
    )
    assert collection.count() == 10
    assert collection.get(ids=["q0"])["documents"] == ["Updated answer"]


def test_retrieves_each_questions_own_answer(workflow):
    retrieve = workflow.namespace["retrieve"]
    for question, answer in workflow.namespace["qa"].items():
        hit = retrieve(question)
        assert hit["question"] == question
        assert hit["answer"] == answer
        assert hit["similarity"] == pytest.approx(1.0)


def test_returns_closest_record_and_converts_cosine_distance(workflow):
    hit = workflow.namespace["retrieve"]("partially similar question")
    assert hit["question"] == next(iter(workflow.namespace["qa"]))
    assert hit["similarity"] == pytest.approx(0.8)


def test_generation_decodes_only_new_tokens(workflow):
    answer = workflow.namespace["generate_answer"]("Example?", "Retrieved context")
    assert answer == "Generated answer"
    assert workflow.decoded[-1] == [99, 100]
    message = workflow.prompts[-1][0]
    assert message["role"] == "user"
    assert "Retrieved context" in message["content"]
    assert "Example?" in message["content"]
    assert workflow.model_name == "meta-llama/Llama-3.2-1B-Instruct"
    assert workflow.tokenizer_name == workflow.model_name


def test_evaluation_saves_baseline_and_rag_for_all_questions(workflow):
    qa = workflow.namespace["qa"]
    assert len(workflow.saved_results) == len(qa)
    for index, result in enumerate(workflow.saved_results):
        assert result["expected"] == qa[result["question"]]
        assert result["context"] == (
            f"Question: {result['question']}\nAnswer: {result['expected']}"
        )
        assert result["without_rag"] == "Generated answer"
        assert result["rag"] == "Generated answer"
        baseline = workflow.prompts[2 * index][0]["content"]
        rag = workflow.prompts[2 * index + 1][0]["content"]
        assert result["context"] not in baseline
        assert result["context"] in rag


def test_storage_paths_are_relative_to_repository(workflow):
    repo = Path(__file__).resolve().parents[1]
    assert workflow.database_paths == [repo / "data" / "chroma"]
    assert workflow.results_path == repo / "results" / "vectordb_results.json"
