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
    GGUF_QUANTIZATION,
    INSTRUCTION_PART,
    LORA_RANK,
    MAX_SEQ_LENGTH,
    RESPONSE_PART,
    answer,
    attach_lora,
    finetune,
    load_base_model,
    render,
    save_adapter,
    save_gguf,
    save_merged,
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


# --- the settings that define the experiment ----------------------------------
#
# Everything below is a number that would change what I trained without
# changing whether the code runs. The two tests above only check the arguments
# I pass in by hand, which means every default was unguarded.


def test_finetune_defaults_are_the_numbers_i_actually_trained_on(monkeypatch):
    """Deliberately written against literals rather than the constants.

    Comparing the config to `BATCH_SIZE` would pass no matter what I set
    `BATCH_SIZE` to, which guards the wiring and nothing else. These are the
    numbers behind the 6 out of 10 in `answers.ipynb`, so changing one should
    mean coming here and changing it on purpose.
    """
    recorder = {}
    fake_unsloth(monkeypatch, recorder)
    fake_training_libraries(monkeypatch, recorder)

    finetune("model", FakeTokenizer(), [{"messages": []}])

    config = recorder["config"]
    assert config["num_train_epochs"] == 3
    assert config["learning_rate"] == 2e-4
    assert config["per_device_train_batch_size"] == 2
    assert config["gradient_accumulation_steps"] == 4
    assert config["max_seq_length"] == 1024


def test_the_constants_are_what_finetune_passes_through(monkeypatch):
    """The other half: the batch settings come from the constants.

    Writing this taught me the two kinds of setting in here aren't the same.
    `BATCH_SIZE` and `GRAD_ACCUM` are read out of the module when the function
    runs, so patching them works. `epochs` is a default argument, so it's
    bound to whatever `EPOCHS` was at import and patching the module does
    nothing. The epochs path is covered by passing the argument instead.
    """
    recorder = {}
    fake_unsloth(monkeypatch, recorder)
    fake_training_libraries(monkeypatch, recorder)
    monkeypatch.setattr("src.trainer.BATCH_SIZE", 7)
    monkeypatch.setattr("src.trainer.GRAD_ACCUM", 9)

    finetune("model", FakeTokenizer(), [{"messages": []}])

    assert recorder["config"]["per_device_train_batch_size"] == 7
    assert recorder["config"]["gradient_accumulation_steps"] == 9


def test_finetune_is_seeded(monkeypatch):
    """Two runs of the same data should give the same model."""
    recorder = {}
    fake_unsloth(monkeypatch, recorder)
    fake_training_libraries(monkeypatch, recorder)
    finetune("model", FakeTokenizer(), [{"messages": []}])
    assert recorder["config"]["seed"] == 3407


def test_finetune_keeps_the_optimizer_and_schedule_from_the_notebook(monkeypatch):
    recorder = {}
    fake_unsloth(monkeypatch, recorder)
    fake_training_libraries(monkeypatch, recorder)
    finetune("model", FakeTokenizer(), [{"messages": []}])
    assert recorder["config"]["optim"] == "adamw_8bit"
    assert recorder["config"]["lr_scheduler_type"] == "linear"
    assert recorder["config"]["warmup_steps"] == 5


def test_finetune_does_not_report_anywhere(monkeypatch):
    # Left on, TRL tries to reach Weights and Biases and the Modal run hangs
    # waiting for an API key I never set.
    recorder = {}
    fake_unsloth(monkeypatch, recorder)
    fake_training_libraries(monkeypatch, recorder)
    finetune("model", FakeTokenizer(), [{"messages": []}])
    assert recorder["config"]["report_to"] == "none"


def test_base_model_is_the_1b_and_not_the_paraphraser():
    """`gpu/paraphrase_app.py` loads the 3B. These are easy to mix up."""
    assert "1B" in BASE_MODEL
    assert "3B" not in BASE_MODEL


def test_load_base_model_passes_the_sequence_length(monkeypatch):
    recorder = fake_unsloth(monkeypatch, {})
    load_base_model()
    assert recorder["from_pretrained"]["max_seq_length"] == MAX_SEQ_LENGTH


def test_attach_lora_keeps_alpha_equal_to_rank(monkeypatch):
    recorder = fake_unsloth(monkeypatch, {})
    attach_lora("model", rank=32)
    assert recorder["get_peft_model"]["lora_alpha"] == 32


def test_attach_lora_turns_dropout_off_and_checkpointing_on(monkeypatch):
    # Dropout would make the run unrepeatable. Gradient checkpointing is what
    # keeps a 1B plus LoRA inside the 14.6GB a T4 gives me.
    recorder = fake_unsloth(monkeypatch, {})
    attach_lora("model")
    assert recorder["get_peft_model"]["lora_dropout"] == 0
    assert recorder["get_peft_model"]["use_gradient_checkpointing"] == "unsloth"


def test_render_keeps_a_system_turn(monkeypatch):
    rows = [
        {
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "hi"},
            ]
        }
    ]
    assert render(rows, FakeTokenizer()) == ["<system>Be brief.<user>hi"]


def test_answer_moves_the_inputs_to_the_device():
    """The real run passes cuda. CPU is the same code path."""
    seen = {}

    class Watching(FakeTokenizer):
        def __call__(self, text, return_tensors=None, add_special_tokens=True):
            out = super().__call__(text, return_tensors, add_special_tokens)
            seen["before"] = out["input_ids"].device.type
            return out

    model = FakeModel()
    answer(model, Watching(), ["a?"], device="cpu")
    assert seen["before"] == "cpu"


def test_answer_only_decodes_the_new_tokens():
    """Decoding the prompt back would put the question inside the answer."""
    captured = {}

    class Watching(FakeTokenizer):
        def decode(self, ids, skip_special_tokens=False):
            captured["length"] = len(ids)
            return "reply"

    # The fake prompt is 3 tokens and the fake generation is 5, so the reply
    # the notebook shows should be the 2 tokens on the end.
    answer(FakeModel(), Watching(), ["a?"], device="cpu")
    assert captured["length"] == 2


# --- saving, which is the thing I forgot twice ------------------------------


class FakeSavable:
    """Records what it was asked to save rather than writing anything."""

    def __init__(self):
        self.calls = {}

    def save_pretrained(self, path):
        self.calls["save_pretrained"] = path

    def save_pretrained_gguf(self, path, tokenizer, quantization_method=None):
        self.calls["gguf"] = (path, tokenizer, quantization_method)

    def save_pretrained_merged(self, path, tokenizer, save_method=None):
        self.calls["merged"] = (path, tokenizer, save_method)


def test_save_adapter_writes_the_model_and_the_tokenizer():
    """The tokenizer matters as much as the weights.

    A LoRA saved without its tokenizer loads, and then produces nonsense,
    because the chat template it was trained against is gone.
    """
    model, tokenizer = FakeSavable(), FakeSavable()
    out = save_adapter(model, tokenizer, "/models/adapter")
    assert model.calls["save_pretrained"] == "/models/adapter"
    assert tokenizer.calls["save_pretrained"] == "/models/adapter"
    assert out == "/models/adapter"


def test_save_gguf_defaults_to_the_four_bit_quantization():
    model, tokenizer = FakeSavable(), object()
    save_gguf(model, tokenizer, "/models/gguf")
    path, tok, quant = model.calls["gguf"]
    assert (path, tok) == ("/models/gguf", tokenizer)
    assert quant == GGUF_QUANTIZATION == "q4_k_m"


def test_save_gguf_quantization_is_adjustable():
    model = FakeSavable()
    save_gguf(model, object(), "/models/gguf", quantization="q8_0")
    assert model.calls["gguf"][2] == "q8_0"


def test_save_merged_asks_for_16_bit():
    # The fallback has to be unquantized, otherwise it isn't a fallback from
    # the quantized path, it's the same lossy thing by another route.
    model, tokenizer = FakeSavable(), object()
    save_merged(model, tokenizer, "/models/merged")
    assert model.calls["merged"] == ("/models/merged", tokenizer, "merged_16bit")
