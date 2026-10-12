"""Tests for the fine tuning code.

None of these need a GPU. CI runs on a machine with no CUDA device, so
`src/trainer.py` keeps Unsloth and TRL behind function level imports and
everything here either runs on CPU or pushes a fake module into `sys.modules`
before calling in.

Faking the libraries means I'm testing the arguments I pass, not that Unsloth
works. That's the right split. If I get the response template wrong, or the
sampling settings drift, the experiment changes underneath me and nothing
would otherwise complain.
"""

import sys
import types

import pytest
import torch

from src.trainer import (
    BASE_MODEL,
    CHAT_TEMPLATE,
    INSTRUCTION_PART,
    LORA_RANK,
    RESPONSE_PART,
    answer,
    attach_lora,
    finetune,
    load_base_model,
    render,
)


class FakeTokenizer:
    """Renders a conversation the way the real chat template roughly would."""

    eos_token_id = 7

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ):
        parts = [f"<{m['role']}>{m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def decode(self, ids, skip_special_tokens=False):
        return "  the answer  "


class FakeModel:
    def __init__(self):
        self.calls = []

    def generate(self, input_ids=None, **kwargs):
        self.calls.append(kwargs)
        return torch.tensor([[1, 2, 3, 40, 41]])


# --- rendering conversations --------------------------------------------------


def test_render_sends_each_conversation_through_the_template():
    rows = [
        {"messages": [{"role": "user", "content": "hi"}]},
        {"messages": [{"role": "assistant", "content": "hello"}]},
    ]
    assert render(rows, FakeTokenizer()) == ["<user>hi", "<assistant>hello"]


def test_render_of_nothing_is_nothing():
    assert render([], FakeTokenizer()) == []


# --- loading the model --------------------------------------------------------


def fake_unsloth(monkeypatch, recorder):
    module = types.ModuleType("unsloth")
    templates = types.ModuleType("unsloth.chat_templates")

    class FastLanguageModel:
        @staticmethod
        def from_pretrained(**kwargs):
            recorder["from_pretrained"] = kwargs
            return "model", "raw-tokenizer"

        @staticmethod
        def get_peft_model(model, **kwargs):
            recorder["get_peft_model"] = kwargs
            return "peft-model"

    def get_chat_template(tokenizer, chat_template=None):
        recorder["chat_template"] = chat_template
        return f"{tokenizer}+{chat_template}"

    def train_on_responses_only(trainer, instruction_part=None, response_part=None):
        recorder["masking"] = (instruction_part, response_part)
        return trainer

    module.FastLanguageModel = FastLanguageModel
    templates.get_chat_template = get_chat_template
    templates.train_on_responses_only = train_on_responses_only
    monkeypatch.setitem(sys.modules, "unsloth", module)
    monkeypatch.setitem(sys.modules, "unsloth.chat_templates", templates)
    return recorder


def test_load_base_model_asks_for_the_instruct_checkpoint(monkeypatch):
    recorder = fake_unsloth(monkeypatch, {})
    model, _ = load_base_model()
    assert recorder["from_pretrained"]["model_name"] == BASE_MODEL
    assert "Instruct" in BASE_MODEL
    assert model == "model"
    # 4 bit would change what I'm measuring and a 1B fits on a T4 without it.
    assert recorder["from_pretrained"]["load_in_4bit"] is False


def test_load_base_model_puts_the_chat_template_on_the_tokenizer(monkeypatch):
    # Without this the tokenizer formats conversations differently at training
    # time than at answering time, and the fine tune is wasted.
    recorder = fake_unsloth(monkeypatch, {})
    _, tokenizer = load_base_model()
    assert recorder["chat_template"] == CHAT_TEMPLATE
    assert tokenizer == f"raw-tokenizer+{CHAT_TEMPLATE}"


def test_attach_lora_targets_attention_and_mlp(monkeypatch):
    recorder = fake_unsloth(monkeypatch, {})
    attach_lora("model")
    kwargs = recorder["get_peft_model"]
    assert kwargs["r"] == LORA_RANK
    assert "q_proj" in kwargs["target_modules"]
    assert "down_proj" in kwargs["target_modules"]
    assert kwargs["random_state"] == 3407


def test_attach_lora_rank_is_adjustable(monkeypatch):
    recorder = fake_unsloth(monkeypatch, {})
    attach_lora("model", rank=8)
    assert recorder["get_peft_model"]["r"] == 8


# --- training -----------------------------------------------------------------


def fake_training_libraries(monkeypatch, recorder):
    trl = types.ModuleType("trl")

    class SFTConfig:
        def __init__(self, **kwargs):
            recorder["config"] = kwargs

    class SFTTrainer:
        def __init__(self, **kwargs):
            recorder["trainer"] = kwargs

        def train(self):
            return "stats"

    trl.SFTConfig = SFTConfig
    trl.SFTTrainer = SFTTrainer
    monkeypatch.setitem(sys.modules, "trl", trl)

    datasets = types.ModuleType("datasets")

    class Dataset:
        @staticmethod
        def from_dict(mapping):
            recorder["rows"] = mapping
            return "dataset"

    datasets.Dataset = Dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    return recorder


def test_finetune_renders_the_rows_and_passes_the_settings(monkeypatch):
    recorder = {}
    fake_unsloth(monkeypatch, recorder)
    fake_training_libraries(monkeypatch, recorder)

    rows = [{"messages": [{"role": "user", "content": "hi"}]}] * 3
    out = finetune(
        "model",
        FakeTokenizer(),
        rows,
        epochs=2,
        learning_rate=1e-4,
        output_dir="/tmp/x",
    )

    assert out == "stats"
    assert recorder["rows"]["text"] == ["<user>hi"] * 3
    assert recorder["config"]["num_train_epochs"] == 2
    assert recorder["config"]["learning_rate"] == 1e-4
    assert recorder["trainer"]["train_dataset"] == "dataset"


def test_finetune_masks_the_prompt_so_only_answers_are_learned(monkeypatch):
    """If this regresses, the model spends its training learning to write questions."""
    recorder = {}
    fake_unsloth(monkeypatch, recorder)
    fake_training_libraries(monkeypatch, recorder)

    finetune("model", FakeTokenizer(), [{"messages": []}])
    assert recorder["masking"] == (INSTRUCTION_PART, RESPONSE_PART)


def test_the_masking_strings_are_the_llama_3_headers():
    assert INSTRUCTION_PART.startswith("<|start_header_id|>user")
    assert RESPONSE_PART.startswith("<|start_header_id|>assistant")


# --- answering ----------------------------------------------------------------


def test_answer_does_not_sample():
    # The notebook I submit has to produce the same answers if anyone re-runs it.
    model = FakeModel()
    answer(model, FakeTokenizer(), ["what is 2+2?"], device="cpu")
    assert model.calls[0]["do_sample"] is False


def test_answer_returns_one_string_per_question():
    model = FakeModel()
    out = answer(model, FakeTokenizer(), ["a?", "b?", "c?"], device="cpu")
    assert out == ["the answer"] * 3


def test_answer_strips_whitespace_off_the_reply():
    out = answer(FakeModel(), FakeTokenizer(), ["a?"], device="cpu")
    assert out[0] == "the answer"


def test_answer_asks_for_a_generation_prompt():
    """Without it the model gets the question and no cue to reply to it."""
    seen = {}

    class Watching(FakeTokenizer):
        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=False
        ):
            seen["add_generation_prompt"] = add_generation_prompt
            return super().apply_chat_template(
                messages, tokenize, add_generation_prompt
            )

    answer(FakeModel(), Watching(), ["a?"], device="cpu")
    assert seen["add_generation_prompt"] is True


def test_answer_honors_the_token_budget():
    model = FakeModel()
    answer(model, FakeTokenizer(), ["a?"], max_new_tokens=42, device="cpu")
    assert model.calls[0]["max_new_tokens"] == 42


# --- the thing the homework warned about --------------------------------------


def test_importing_the_trainer_does_not_pull_in_unsloth():
    """CI has no GPU, so a top level Unsloth import turns the pipeline red."""
    import importlib

    for name in ("unsloth", "unsloth.chat_templates", "trl"):
        sys.modules.pop(name, None)
    importlib.reload(importlib.import_module("src.trainer"))
    assert "unsloth" not in sys.modules
    assert "trl" not in sys.modules


@pytest.mark.parametrize("name", ["unsloth", "trl", "datasets"])
def test_the_gpu_libraries_are_not_imported_at_module_scope(name):
    source = (
        __import__("pathlib").Path(__file__).parent.parent / "src" / "trainer.py"
    ).read_text()
    header = source.split("def ")[0]
    assert f"import {name}" not in header
