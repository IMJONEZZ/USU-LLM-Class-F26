"""Tests for the fine tuning code.

None of these need a GPU, which is the point. The homework warned that CI would
fall over because the runner has no CUDA device, so `src/trainer.py` keeps
Unsloth and TRL behind function level imports and everything here either runs on
CPU or swaps a fake module into `sys.modules` before calling in.

The fake model just returns a loss I picked, so I can check the perplexity math
against a number I worked out myself instead of against whatever a real model
happened to produce.
"""

import math
import sys
import types

import pytest
import torch

from src.trainer import (
    BASE_MODEL,
    LORA_RANK,
    attach_lora,
    build_splits,
    finetune,
    load_base_model,
    perplexity,
    summarize,
    windows,
)


class FakeOutput:
    def __init__(self, loss):
        self.loss = loss


class FakeModel:
    """Returns a fixed loss so the perplexity math is checkable by hand."""

    def __init__(self, loss=2.0):
        self.loss = loss
        self.eval_called = False
        self.seen = []

    def eval(self):
        self.eval_called = True

    def __call__(self, input_ids=None, labels=None):
        self.seen.append(input_ids)
        return FakeOutput(torch.tensor(self.loss))


class FakeTokenizer:
    """Encodes one token per character so the counts are easy to reason about."""

    def __init__(self, eos_token="<eos>"):
        self.eos_token = eos_token

    def encode(self, text):
        return list(range(len(text)))


def ids(n):
    return torch.arange(n)


# --- chopping the stream into chunks -----------------------------------------


def test_windows_splits_evenly_when_it_divides():
    assert len(windows(ids(100), max_length=10)) == 10


def test_windows_drops_the_short_tail():
    # 95 tokens at length 10 is 9 full chunks and a 5 token remainder that goes
    # in the bin, because a short chunk would make the loss average wrong.
    chunks = windows(ids(95), max_length=10)
    assert len(chunks) == 9
    assert all(len(c) == 10 for c in chunks)


def test_windows_of_a_stream_shorter_than_one_chunk():
    assert windows(ids(5), max_length=10) == []


def test_windows_do_not_overlap():
    chunks = windows(ids(30), max_length=10)
    assert chunks[0][-1].item() == 9
    assert chunks[1][0].item() == 10


# --- perplexity ---------------------------------------------------------------


def test_perplexity_is_exp_of_the_mean_loss():
    model = FakeModel(loss=2.0)
    out = perplexity(model, ids(50), max_length=10)
    assert out["loss"] == pytest.approx(2.0)
    assert out["perplexity"] == pytest.approx(math.exp(2.0))


def test_perplexity_averages_over_every_chunk():
    model = FakeModel(loss=1.5)
    out = perplexity(model, ids(50), max_length=10)
    assert out["chunks"] == 5
    assert out["tokens"] == 50
    assert len(model.seen) == 5


def test_perplexity_puts_the_model_in_eval_mode():
    # Dropout during scoring would make the number different every run.
    model = FakeModel()
    perplexity(model, ids(20), max_length=10)
    assert model.eval_called


def test_perplexity_feeds_batched_input():
    model = FakeModel()
    perplexity(model, ids(20), max_length=10)
    assert model.seen[0].shape == (1, 10)


def test_perplexity_moves_the_chunks_to_the_device(monkeypatch):
    """The real run passes device="cuda". CPU is the same code path."""
    model = FakeModel()
    perplexity(model, ids(20), max_length=10, device="cpu")
    assert model.seen[0].device.type == "cpu"


def test_perplexity_refuses_a_stream_too_short_to_score():
    with pytest.raises(ValueError, match="need at least"):
        perplexity(FakeModel(), ids(5), max_length=10)


def test_lower_loss_means_lower_perplexity():
    worse = perplexity(FakeModel(loss=4.0), ids(20), max_length=10)
    better = perplexity(FakeModel(loss=3.0), ids(20), max_length=10)
    assert better["perplexity"] < worse["perplexity"]


# --- the before and after summary ---------------------------------------------


def test_summarize_reports_the_drop_and_the_percentage():
    before = {"perplexity": 100.0, "loss": 4.6, "chunks": 7, "tokens": 3584}
    after = {"perplexity": 25.0, "loss": 3.2, "chunks": 7, "tokens": 3584}
    out = summarize(before, after)
    assert out["perplexity_drop"] == pytest.approx(75.0)
    assert out["perplexity_drop_pct"] == pytest.approx(75.0)


def test_summarize_shows_a_negative_drop_when_it_got_worse():
    # If training made things worse I want that to come out as a negative
    # number rather than quietly showing up as an improvement.
    before = {"perplexity": 20.0, "loss": 3.0, "chunks": 1, "tokens": 10}
    after = {"perplexity": 25.0, "loss": 3.2, "chunks": 1, "tokens": 10}
    out = summarize(before, after)
    assert out["perplexity_drop"] < 0
    assert out["perplexity_drop_pct"] < 0


def test_summarize_folds_in_the_training_stats():
    before = {"perplexity": 10.0, "loss": 2.3, "chunks": 1, "tokens": 10}
    after = {"perplexity": 5.0, "loss": 1.6, "chunks": 1, "tokens": 10}
    out = summarize(before, after, {"train_runtime": 95.4, "train_loss": 2.58})
    assert out["train_runtime_s"] == 95.4
    assert out["train_loss"] == 2.58


def test_summarize_without_training_stats_leaves_them_out():
    before = {"perplexity": 10.0, "loss": 2.3, "chunks": 1, "tokens": 10}
    after = {"perplexity": 5.0, "loss": 1.6, "chunks": 1, "tokens": 10}
    assert "train_runtime_s" not in summarize(before, after)


# --- the split, using the assignment 2 loader --------------------------------


@pytest.fixture
def dataset_file(tmp_path):
    import json

    rows = [
        {"Character": f"SPEAKER{i}", "Line": f"line number {i}"} for i in range(400)
    ]
    path = tmp_path / "SW_EpisodeIV_VI.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return str(path)


def test_build_splits_returns_two_streams(dataset_file):
    train_ids, val_ids = build_splits(FakeTokenizer(), path=dataset_file, max_length=8)
    assert len(train_ids) > len(val_ids) > 0


def test_build_splits_uses_the_tokenizers_own_eos(dataset_file):
    """The assignment 2 separator isn't a Llama token, so it has to be swapped.

    A longer separator means more characters, and this tokenizer is one token
    per character, so the stream gets longer. That's a cheap way to prove the
    separator actually went through.
    """
    short = build_splits(FakeTokenizer("<a>"), path=dataset_file, max_length=8)
    long = build_splits(FakeTokenizer("<aaaaaaaa>"), path=dataset_file, max_length=8)
    assert len(long[0]) > len(short[0])


def test_build_splits_falls_back_when_there_is_no_eos(dataset_file):
    tokenizer = FakeTokenizer()
    tokenizer.eos_token = None
    train_ids, _ = build_splits(tokenizer, path=dataset_file, max_length=8)
    assert len(train_ids) > 0


def test_val_fraction_controls_the_split(dataset_file):
    train_ids, val_ids = build_splits(
        FakeTokenizer(), path=dataset_file, max_length=8, val_fraction=0.5
    )
    assert abs(len(train_ids) - len(val_ids)) <= 1


# --- the GPU paths, with the GPU libraries faked out -------------------------


def test_importing_the_trainer_does_not_pull_in_unsloth():
    """This is the one the homework warned about.

    CI has no GPU, so if Unsloth were imported at the top of the module the
    whole pipeline would go red before a single test ran.
    """
    import importlib

    for name in ("unsloth", "trl"):
        sys.modules.pop(name, None)
    importlib.reload(importlib.import_module("src.trainer"))
    assert "unsloth" not in sys.modules
    assert "trl" not in sys.modules


def fake_unsloth(monkeypatch, recorder):
    module = types.ModuleType("unsloth")

    class FastLanguageModel:
        @staticmethod
        def from_pretrained(**kwargs):
            recorder["from_pretrained"] = kwargs
            return "model", "tokenizer"

        @staticmethod
        def get_peft_model(model, **kwargs):
            recorder["get_peft_model"] = kwargs
            return "peft-model"

    module.FastLanguageModel = FastLanguageModel
    monkeypatch.setitem(sys.modules, "unsloth", module)
    return module


def test_load_base_model_asks_for_the_right_checkpoint(monkeypatch):
    recorder = {}
    fake_unsloth(monkeypatch, recorder)
    model, tokenizer = load_base_model()
    assert (model, tokenizer) == ("model", "tokenizer")
    assert recorder["from_pretrained"]["model_name"] == BASE_MODEL
    # 4 bit would change what I'm measuring, and a 1B fits without it.
    assert recorder["from_pretrained"]["load_in_4bit"] is False


def test_attach_lora_targets_attention_and_mlp(monkeypatch):
    recorder = {}
    fake_unsloth(monkeypatch, recorder)
    attach_lora("model")
    kwargs = recorder["get_peft_model"]
    assert kwargs["r"] == LORA_RANK
    assert "q_proj" in kwargs["target_modules"]
    assert "down_proj" in kwargs["target_modules"]
    assert kwargs["random_state"] == 3407


def test_attach_lora_rank_is_adjustable(monkeypatch):
    recorder = {}
    fake_unsloth(monkeypatch, recorder)
    attach_lora("model", rank=8)
    assert recorder["get_peft_model"]["r"] == 8


def test_finetune_builds_a_text_dataset_and_passes_the_settings(monkeypatch):
    recorder = {}

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

    class Tok:
        def decode(self, chunk):
            return "text"

    out = finetune(
        "model", Tok(), ids(2048), max_steps=7, learning_rate=1e-4, output_dir="/tmp/x"
    )

    assert out == "stats"
    assert recorder["config"]["max_steps"] == 7
    assert recorder["config"]["learning_rate"] == 1e-4
    assert recorder["trainer"]["train_dataset"] == "dataset"
    # 2048 tokens at 512 is four chunks, so four training rows.
    assert len(recorder["rows"]["text"]) == 4
