"""Unit tests for the RAG workflow without launching Modal or downloading models."""

from unittest.mock import Mock

import torch

from src import vectordb
from src.modal import modal_rag


class FakeBatchEncoding(dict):
    """Minimal tokenizer output that supports moving tensors to a device."""

    def to(self, device):
        self["input_ids"] = self["input_ids"].to(device)
        return self


class FakeTokenizer:
    def __init__(self):
        self.prompt = None
        self.options = None

    def __call__(self, prompt, **options):
        self.prompt = prompt
        self.options = options
        return FakeBatchEncoding(input_ids=torch.tensor([[1, 2]]))

    def decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens is True
        assert tokens.tolist() == [3, 4]
        return "  generated answer\n"


class FakeModel:
    def __init__(self):
        self.options = None

    def generate(self, **options):
        self.options = options
        return torch.tensor([[1, 2, 3, 4]])


def test_initialize_components_uses_expected_models(monkeypatch):
    import sentence_transformers
    import transformers

    embedding_model = object()
    collection = object()
    tokenizer = object()
    model = Mock()
    model.to.return_value = model

    sentence_transformer_factory = Mock(return_value=embedding_model)
    tokenizer_factory = Mock()
    tokenizer_factory.from_pretrained.return_value = tokenizer
    causal_model_factory = Mock()
    causal_model_factory.from_pretrained.return_value = model
    create_collection = Mock(return_value=collection)

    monkeypatch.setattr(
        sentence_transformers, "SentenceTransformer", sentence_transformer_factory
    )
    monkeypatch.setattr(transformers, "AutoTokenizer", tokenizer_factory)
    monkeypatch.setattr(transformers, "AutoModelForCausalLM", causal_model_factory)
    monkeypatch.setattr(vectordb, "create_collection", create_collection)

    components = modal_rag.initialize_components()

    assert components == (embedding_model, collection, tokenizer, model)
    sentence_transformer_factory.assert_called_once_with(
        vectordb.MODEL_NAME, device="cpu"
    )
    create_collection.assert_called_once_with(embedding_model, vectordb.documents)
    tokenizer_factory.from_pretrained.assert_called_once_with(modal_rag.LLAMA_MODEL_ID)
    causal_model_factory.from_pretrained.assert_called_once_with(
        modal_rag.LLAMA_MODEL_ID,
        torch_dtype=torch.float16,
    )
    model.to.assert_called_once_with("cuda")
    model.eval.assert_called_once_with()


def test_build_prompt_includes_context_question_and_chat_tokens():
    prompt = modal_rag.build_prompt("retrieved text", "the question")

    assert "<|begin_of_text|>" in prompt
    assert "<|start_header_id|>system<|end_header_id|>" in prompt
    assert "<|start_header_id|>user<|end_header_id|>" in prompt
    assert "<|eot_id|>" in prompt
    assert "retrieved text\nthe question" in prompt
    assert prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")


def test_answer_question_uses_retrieved_context(monkeypatch):
    embedding_model = object()
    collection = object()
    tokenizer = object()
    model = object()
    query = Mock(return_value=[{"document": "retrieved text"}])
    generate = Mock(return_value="answer")
    monkeypatch.setattr(vectordb, "query_collection", query)
    monkeypatch.setattr(modal_rag, "generate_answer", generate)

    result = modal_rag.answer_question(
        "the question", embedding_model, collection, tokenizer, model
    )

    assert result == "answer"
    query.assert_called_once_with(
        embedding_model,
        collection,
        "the question",
        n_results=1,
    )
    prompt = generate.call_args.args[0]
    assert "retrieved text\nthe question" in prompt
    assert generate.call_args.args[1:] == (tokenizer, model)


def test_generate_answer_decodes_only_new_tokens():
    tokenizer = FakeTokenizer()
    model = FakeModel()

    answer = modal_rag.generate_answer(
        "formatted prompt", tokenizer, model, device="cpu"
    )

    assert answer == "generated answer"
    assert tokenizer.prompt == "formatted prompt"
    assert tokenizer.options == {
        "return_tensors": "pt",
        "add_special_tokens": False,
    }
    assert model.options["max_new_tokens"] == 256
    assert model.options["do_sample"] is False
