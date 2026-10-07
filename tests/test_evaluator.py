"""Tests for ROUGE evaluation on Episode VI continuations."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src import evaluator as evaluator_module


def test_make_continuation_examples_splits_tokens_in_half():
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5, 6]
    tokenizer.decode.side_effect = lambda tokens, skip_special_tokens: str(tokens)

    prompts, references = evaluator_module.make_continuation_examples(
        ["dialogue"], tokenizer
    )

    assert prompts == ["[1, 2, 3]"]
    assert references == ["[4, 5, 6]"]


def test_make_continuation_examples_skips_very_short_lines():
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]

    assert evaluator_module.make_continuation_examples(["short"], tokenizer) == (
        [],
        [],
    )


def test_run_evaluation_uses_episode_six_and_returns_rouge(monkeypatch):
    source = [
        {"film": "ep5_empire_strikes_back", "speaker": "Luke", "text": "not test"},
        {
            "film": "ep6_return_of_the_jedi",
            "speaker": "Luke",
            "text": "Luke is a Jedi Knight now.",
        },
    ]
    tokenizer = MagicMock()
    tokenizer.pad_token = "<eos>"
    tokenizer.pad_token_id = 0
    tokenizer.encode.return_value = [1, 2, 3, 4, 5, 6]
    tokenizer.decode.side_effect = lambda tokens, skip_special_tokens: str(tokens)
    generation = MagicMock(return_value=[[{"generated_text": "continuation"}]])
    base_model = MagicMock()
    adapted_model = MagicMock()
    metric = MagicMock()
    metric.compute.return_value = {"rouge1": 0.5, "rouge2": 0.2, "rougeL": 0.4}
    load_dataset = MagicMock(return_value=source)
    evaluate_module = SimpleNamespace(load=MagicMock(return_value=metric))
    datasets_module = SimpleNamespace(load_dataset=load_dataset)
    transformers_module = SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(
            from_pretrained=MagicMock(return_value=base_model)
        ),
        AutoTokenizer=SimpleNamespace(
            from_pretrained=MagicMock(return_value=tokenizer)
        ),
        pipeline=MagicMock(return_value=generation),
    )
    peft_module = SimpleNamespace(
        PeftModel=SimpleNamespace(from_pretrained=MagicMock(return_value=adapted_model))
    )
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)
    monkeypatch.setitem(sys.modules, "evaluate", evaluate_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "peft", peft_module)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    result = evaluator_module.run_evaluation(
        "meta-llama/Llama-3.2-1B", adapter_path="adapter-directory"
    )

    assert result["rouge1"] == 0.5
    load_dataset.assert_called_once_with(
        "IMJONEZZ/star-wars-dataset", "cues", split="train"
    )
    transformers_module.pipeline.assert_called_once()
    peft_module.PeftModel.from_pretrained.assert_called_once_with(
        base_model, "adapter-directory"
    )
    assert transformers_module.pipeline.call_args.kwargs["model"] is adapted_model
    metric.compute.assert_called_once_with(
        predictions=["continuation"],
        references=["[4, 5, 6]"],
        use_stemmer=True,
    )


def test_run_evaluation_rejects_non_test_split():
    with pytest.raises(ValueError, match="Episode VI"):
        evaluator_module.run_evaluation("trained-model", split="validation")
