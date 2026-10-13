import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from src import config, data, train
from tests.helpers import (
    LONG_Q,
    _evaluator,
    _fake_unsloth,
    _HFStyleTokenizer,
    _teacher_chain,
    _tiny_tokenizer,
    _trow,
    _without_line,
)


# ---- M6: train.py (memory estimate and GPU choice) ----
def test_memory_estimate_has_positive_parts_that_sum_to_the_total():
    parts = train.estimate_training_memory_gib(256, batch_size=1)
    pieces = ("weights", "lora_and_optimizer", "activations", "logits", "overhead")
    assert all(parts[name] > 0 for name in pieces)
    assert parts["total"] == pytest.approx(sum(parts[name] for name in pieces))


def test_memory_estimate_grows_with_tokens_and_batch_size():
    small = train.estimate_training_memory_gib(256, batch_size=1)["total"]
    longer = train.estimate_training_memory_gib(1024, batch_size=1)["total"]
    wider = train.estimate_training_memory_gib(256, batch_size=4)["total"]
    assert small < longer
    assert small < wider


def test_memory_estimate_logits_scale_linearly_with_tokens():
    short = train.estimate_training_memory_gib(256)["logits"]
    long = train.estimate_training_memory_gib(512)["logits"]
    assert long == pytest.approx(2 * short)


def test_memory_estimate_for_short_examples_is_a_few_gib():
    assert 1 < train.estimate_training_memory_gib(256)["total"] < 8


@pytest.mark.parametrize("tokens, batch", [(0, 1), (-1, 1), (256, 0), (256, -2)])
def test_memory_estimate_rejects_non_positive_inputs(tokens, batch):
    with pytest.raises(ValueError):
        train.estimate_training_memory_gib(tokens, batch_size=batch)


OPTIONS = {
    "A": {"vram_gib": 16, "usd_per_second": 0.3},
    "B": {"vram_gib": 24, "usd_per_second": 0.2},
    "C": {"vram_gib": 48, "usd_per_second": 0.5},
}


def test_choose_gpu_picks_the_cheapest_that_fits_not_the_smallest():
    assert train.choose_gpu(10, OPTIONS) == "B"


def test_choose_gpu_only_trusts_a_fraction_of_the_memory():
    assert train.choose_gpu(20, OPTIONS) == "B"  # 24 * 0.85 = 20.4
    assert train.choose_gpu(21, OPTIONS) == "C"


def test_choose_gpu_with_nothing_big_enough_raises():
    with pytest.raises(ValueError, match="fits"):
        train.choose_gpu(100, OPTIONS)


def test_choose_gpu_breaks_a_price_tie_with_the_smaller_memory():
    tied = {
        "big": {"vram_gib": 48, "usd_per_second": 0.2},
        "small": {"vram_gib": 24, "usd_per_second": 0.2},
    }
    assert train.choose_gpu(5, tied) == "small"


@pytest.mark.parametrize(
    ("needed", "expected"), [(5, "T4"), (20, "L4"), (30, "L40S"), (60, "A100-80GB")]
)
def test_choose_gpu_with_the_real_modal_prices(needed, expected):
    assert train.choose_gpu(needed) == expected


@pytest.mark.parametrize(
    ("n", "batch", "accum", "expected"),
    [
        (383, 4, 4, 24),
        (450, 4, 4, 29),
        (3, 4, 4, 1),
        (64, 4, 4, 4),
        (65, 4, 4, 5),
        (1, 1, 1, 1),
    ],
)
def test_steps_per_epoch_rounds_up_and_is_never_zero(n, batch, accum, expected):
    assert train.steps_per_epoch(n, batch, accum) == expected


def test_total_steps_is_steps_per_epoch_times_epochs():
    assert train.total_steps(383, 4, 4, 3) == 72


@pytest.mark.parametrize("args", [(0, 4, 4), (10, 0, 4), (10, 4, 0), (-1, 4, 4)])
def test_steps_per_epoch_rejects_non_positive_inputs(args):
    with pytest.raises(ValueError):
        train.steps_per_epoch(*args)


def test_the_training_hyperparameters_are_the_chosen_ones():
    assert config.PER_DEVICE_TRAIN_BATCH_SIZE == 4
    assert config.GRADIENT_ACCUMULATION_STEPS == 4
    assert config.EARLY_STOPPING is True
    assert config.EARLY_STOPPING_PATIENCE == 1
    assert config.EPOCHS == 3


def test_clean_overrides_drops_none_but_keeps_zero_and_false():
    cleaned = train.clean_overrides(
        {"seed": 0, "flag": False, "epochs": None, "lr": 1e-4}
    )
    assert cleaned == {"seed": 0, "flag": False, "lr": 1e-4}


def test_hyperparameters_cover_the_settings_that_change_a_run():
    h = train.hyperparameters()
    for key in (
        "model",
        "learning_rate",
        "epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "lora_r",
        "lora_alpha",
        "lora_target_modules",
        "max_seq_length",
        "seed",
        "early_stopping",
        "early_stopping_patience",
    ):
        assert key in h
    assert h["learning_rate"] == config.LEARNING_RATE


def test_hyperparameter_hash_is_stable_and_changes_with_any_setting():
    base = train.hyperparameters()
    assert train.hyperparameter_hash(base) == train.hyperparameter_hash(
        train.hyperparameters()
    )
    assert train.hyperparameter_hash(base) != train.hyperparameter_hash(
        train.hyperparameters(learning_rate=1e-4)
    )
    assert train.hyperparameter_hash(base) != train.hyperparameter_hash(
        train.hyperparameters(epochs=5)
    )
    assert len(train.hyperparameter_hash(base)) == 64


def test_hyperparameter_overrides_must_name_a_known_setting():
    with pytest.raises(ValueError, match="learning_rte"):
        train.hyperparameters(learning_rte=1e-4)


# ---- M8: the text dataset ----
def test_build_text_dataset_formats_every_row_and_ends_with_the_end_token():
    rows = [_trow(1), _trow(2, chain=_teacher_chain("9", "other words"))]
    dataset = train.build_text_dataset(rows, config.EOS_TOKEN)
    assert list(dataset["text"]) == [
        data.format_example(r["question"], r["chain"], config.EOS_TOKEN) for r in rows
    ]
    assert all(text.endswith(config.EOS_TOKEN) for text in dataset["text"])


def test_build_text_dataset_rejects_an_empty_response_naming_the_row():
    with pytest.raises(ValueError, match="t-2"):
        train.build_text_dataset([_trow(1), _trow(2, chain="")], config.EOS_TOKEN)


def test_build_text_dataset_rejects_a_chain_with_a_missing_section_naming_the_row():
    broken = _without_line(_teacher_chain("9"), "Verify:")
    with pytest.raises(ValueError, match="t-1"):
        train.build_text_dataset([_trow(1, chain=broken)], config.EOS_TOKEN)


def test_build_text_dataset_with_no_rows_raises():
    with pytest.raises(ValueError, match="no rows"):
        train.build_text_dataset([], config.EOS_TOKEN)


def test_load_model_for_training_loads_the_base_model_in_4bit(monkeypatch):
    fast = _fake_unsloth(monkeypatch)
    train.load_model_for_training()
    kwargs = fast.from_pretrained.call_args.kwargs
    assert kwargs["model_name"] == config.MODEL_NAME
    assert kwargs["load_in_4bit"] is True
    assert kwargs["max_seq_length"] == config.MAX_SEQ_LENGTH


def test_load_model_for_training_attaches_lora_with_the_configured_settings(
    monkeypatch,
):
    fast = _fake_unsloth(monkeypatch)
    result = train.load_model_for_training()
    kwargs = fast.get_peft_model.call_args.kwargs
    assert fast.get_peft_model.call_args.args == ("base_model",)
    assert kwargs["r"] == config.LORA_R
    assert kwargs["lora_alpha"] == config.LORA_ALPHA
    assert kwargs["target_modules"] == config.LORA_TARGET_MODULES
    assert kwargs["lora_dropout"] == 0
    assert kwargs["bias"] == "none"
    assert kwargs["use_gradient_checkpointing"] == "unsloth"
    assert kwargs["random_state"] == config.SEED
    assert result == ("peft_model", "tokenizer")


def test_load_model_for_training_does_not_switch_to_inference_mode(monkeypatch):
    fast = _fake_unsloth(monkeypatch)
    train.load_model_for_training()
    fast.for_inference.assert_not_called()


def test_save_adapter_saves_the_model_and_tokenizer():
    trainer, tokenizer = MagicMock(), MagicMock()
    train.save_adapter(trainer, tokenizer, "some/dir")
    trainer.save_model.assert_called_once_with("some/dir")
    tokenizer.save_pretrained.assert_called_once_with("some/dir")


def _write_adapter(root, round_n, base="unsloth/Llama-3.2-1B", rank=config.LORA_R):
    folder = config.adapter_path(round_n, root=root)
    folder.mkdir(parents=True)
    (folder / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base, "r": rank}), encoding="utf-8"
    )
    return folder


def test_load_adapter_returns_the_latest_rounds_folder(tmp_path):
    _write_adapter(tmp_path, 0)
    _write_adapter(tmp_path, 1)
    assert (
        train.load_adapter([{"round": 0}, {"round": 1}], root=tmp_path)
        == tmp_path / "round_1"
    )


def test_load_adapter_with_a_missing_folder_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="round_0"):
        train.load_adapter([{"round": 0}], root=tmp_path)


def test_load_adapter_with_a_folder_but_no_config_raises(tmp_path):
    (tmp_path / "round_0").mkdir()
    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        train.load_adapter([{"round": 0}], root=tmp_path)


def test_load_adapter_from_another_base_model_raises(tmp_path):
    _write_adapter(tmp_path, 0, base="meta-llama/Llama-3.1-8B")
    with pytest.raises(ValueError, match="base model"):
        train.load_adapter([{"round": 0}], root=tmp_path)


@pytest.mark.parametrize(
    "base",
    ["unsloth/Llama-3.2-1B", "unsloth/llama-3.2-1b-bnb-4bit", "UNSLOTH/LLAMA-3.2-1B"],
)
def test_load_adapter_accepts_the_same_base_model_under_its_usual_names(tmp_path, base):
    _write_adapter(tmp_path, 0, base=base)
    assert train.load_adapter([{"round": 0}], root=tmp_path) == tmp_path / "round_0"


def test_load_adapter_with_a_different_rank_raises(tmp_path):
    _write_adapter(tmp_path, 0, rank=config.LORA_R * 2)
    with pytest.raises(ValueError, match="rank"):
        train.load_adapter([{"round": 0}], root=tmp_path)


def test_load_adapter_with_a_gap_in_the_round_numbers_raises(tmp_path):
    _write_adapter(tmp_path, 0)
    _write_adapter(tmp_path, 2)
    with pytest.raises(ValueError, match="gap"):
        train.load_adapter([{"round": 0}, {"round": 2}], root=tmp_path)


def test_load_adapter_with_no_rounds_recorded_raises(tmp_path):
    with pytest.raises(ValueError, match="no rounds"):
        train.load_adapter([], root=tmp_path)


# ---- M8: the trainer (trl, transformers and unsloth are faked) ----
def _fake_training_stack(monkeypatch):
    trl = ModuleType("trl")
    trl.SFTConfig = MagicMock(name="SFTConfig")
    trl.SFTTrainer = MagicMock(name="SFTTrainer")
    transformers = ModuleType("transformers")
    transformers.EarlyStoppingCallback = MagicMock(name="EarlyStoppingCallback")
    chat = ModuleType("unsloth.chat_templates")
    chat.train_on_responses_only = MagicMock(
        name="train_on_responses_only", return_value="masked_trainer"
    )
    unsloth = ModuleType("unsloth")
    unsloth.chat_templates = chat
    for name, module in (
        ("trl", trl),
        ("transformers", transformers),
        ("unsloth", unsloth),
        ("unsloth.chat_templates", chat),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return trl, transformers, chat


def _datasets(n_train=3, n_eval=2):
    return ["train"] * n_train, ["eval"] * n_eval


def _trainer_with_fakes(monkeypatch, **overrides):
    trl, transformers, chat = _fake_training_stack(monkeypatch)
    train_ds, eval_ds = overrides.pop("datasets", _datasets())
    result = train.build_trainer("model", "tok", train_ds, eval_ds, "out", **overrides)
    return trl, transformers, chat, result


def test_build_trainer_configures_epochs_batching_and_evaluation_per_epoch(monkeypatch):
    trl, _, _, _ = _trainer_with_fakes(monkeypatch)
    kwargs = trl.SFTConfig.call_args.kwargs
    assert kwargs["output_dir"] == "out"
    assert kwargs["num_train_epochs"] == config.EPOCHS
    assert kwargs["per_device_train_batch_size"] == config.PER_DEVICE_TRAIN_BATCH_SIZE
    assert kwargs["per_device_eval_batch_size"] == config.PER_DEVICE_EVAL_BATCH_SIZE
    assert kwargs["gradient_accumulation_steps"] == config.GRADIENT_ACCUMULATION_STEPS
    assert kwargs["learning_rate"] == config.LEARNING_RATE
    assert kwargs["eval_strategy"] == "epoch"
    assert kwargs["save_strategy"] == "epoch"
    assert kwargs["logging_strategy"] == "epoch"
    assert kwargs["load_best_model_at_end"] is True
    assert kwargs["metric_for_best_model"] == "eval_loss"
    assert kwargs["greater_is_better"] is False
    assert kwargs["dataset_text_field"] == "text"
    assert kwargs["max_length"] == config.MAX_SEQ_LENGTH
    assert kwargs["seed"] == config.SEED
    assert kwargs["report_to"] == "none"


def test_build_trainer_never_truncates_below_the_audited_limit(monkeypatch):
    trl, _, _, _ = _trainer_with_fakes(monkeypatch)
    assert trl.SFTConfig.call_args.kwargs["max_length"] >= 296


def test_build_trainer_accepts_overrides_including_a_zero_seed(monkeypatch):
    trl, _, _, _ = _trainer_with_fakes(
        monkeypatch,
        epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        seed=0,
    )
    kwargs = trl.SFTConfig.call_args.kwargs
    assert (
        kwargs["num_train_epochs"],
        kwargs["per_device_train_batch_size"],
        kwargs["gradient_accumulation_steps"],
        kwargs["seed"],
    ) == (1, 2, 8, 0)


def test_build_trainer_passes_the_model_data_and_tokenizer(monkeypatch):
    trl, _, _, _ = _trainer_with_fakes(monkeypatch)
    kwargs = trl.SFTTrainer.call_args.kwargs
    assert kwargs["model"] == "model"
    assert kwargs["processing_class"] == "tok"
    assert kwargs["args"] is trl.SFTConfig.return_value
    assert len(kwargs["train_dataset"]) == 3 and len(kwargs["eval_dataset"]) == 2


def test_build_trainer_early_stopping_on_adds_one_callback_with_the_patience(
    monkeypatch,
):
    trl, transformers, _, _ = _trainer_with_fakes(
        monkeypatch, early_stopping=True, early_stopping_patience=2
    )
    transformers.EarlyStoppingCallback.assert_called_once_with(
        early_stopping_patience=2
    )
    assert trl.SFTTrainer.call_args.kwargs["callbacks"] == [
        transformers.EarlyStoppingCallback.return_value
    ]


def test_build_trainer_early_stopping_off_adds_no_stopping_callback(monkeypatch):
    trl, transformers, _, _ = _trainer_with_fakes(monkeypatch, early_stopping=False)
    transformers.EarlyStoppingCallback.assert_not_called()
    assert trl.SFTTrainer.call_args.kwargs["callbacks"] == []


def test_build_trainer_uses_the_configured_early_stopping_by_default(monkeypatch):
    _, transformers, _, _ = _trainer_with_fakes(monkeypatch)
    transformers.EarlyStoppingCallback.assert_called_once_with(
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE
    )


def test_build_trainer_appends_extra_callbacks_after_early_stopping(monkeypatch):
    extra = object()
    trl, transformers, _, _ = _trainer_with_fakes(monkeypatch, extra_callbacks=[extra])
    assert trl.SFTTrainer.call_args.kwargs["callbacks"] == [
        transformers.EarlyStoppingCallback.return_value,
        extra,
    ]


def test_build_trainer_masks_everything_before_the_response(monkeypatch):
    trl, _, chat, result = _trainer_with_fakes(monkeypatch)
    chat.train_on_responses_only.assert_called_once_with(
        trl.SFTTrainer.return_value,
        instruction_part=config.INSTRUCTION_MARKER,
        response_part=config.RESPONSE_MARKER,
    )
    assert result == "masked_trainer"


def test_build_trainer_with_an_empty_eval_set_raises(monkeypatch):
    with pytest.raises(ValueError, match="eval dataset is empty"):
        _trainer_with_fakes(monkeypatch, datasets=(["t"], []))


def test_build_trainer_with_an_empty_train_set_raises(monkeypatch):
    with pytest.raises(ValueError, match="train dataset is empty"):
        _trainer_with_fakes(monkeypatch, datasets=([], ["e"]))


def test_epoch_evaluator_records_the_epoch_losses_and_scores(monkeypatch):
    fast = _fake_unsloth(monkeypatch)
    evaluator = _evaluator()
    record = evaluator.evaluate_epoch("model", epoch=1, train_loss=1.5, eval_loss=1.2)
    assert record["epoch"] == 1
    assert (record["train_loss"], record["eval_loss"]) == (1.5, 1.2)
    assert record["accuracy"] == 0.5
    assert record["format_compliance"] == 1.0
    assert evaluator.epochs == [record]
    fast.for_inference.assert_called_once_with("model")
    fast.for_training.assert_called_once_with("model")


def test_epoch_evaluator_restores_training_mode_even_if_generation_throws(monkeypatch):
    fast = _fake_unsloth(monkeypatch)
    evaluator = _evaluator(raise_in_generate=True)
    with pytest.raises(RuntimeError, match="generation failed"):
        evaluator.evaluate_epoch("model", 1, 1.0, 1.0)
    fast.for_training.assert_called_once_with("model")
    assert evaluator.epochs == []


def test_epoch_evaluator_accepts_missing_losses(monkeypatch):
    _fake_unsloth(monkeypatch)
    record = _evaluator().evaluate_epoch("m", 1, train_loss=None, eval_loss=None)
    assert (record["train_loss"], record["eval_loss"]) == (None, None)


def test_epoch_evaluator_with_no_eval_rows_raises():
    with pytest.raises(ValueError, match="eval rows"):
        train.EpochEvaluator([], lambda prompts: [])


def _fake_transformers_callback(monkeypatch):
    module = ModuleType("transformers")

    class TrainerCallback:
        pass

    module.TrainerCallback = TrainerCallback
    monkeypatch.setitem(sys.modules, "transformers", module)
    return TrainerCallback


def test_the_callback_runs_the_evaluator_at_each_evaluation_with_the_epoch_and_losses(
    monkeypatch,
):
    _fake_unsloth(monkeypatch)
    base = _fake_transformers_callback(monkeypatch)
    evaluator = _evaluator()
    callback = train.make_epoch_eval_callback(evaluator)
    assert isinstance(callback, base)
    state = SimpleNamespace(
        epoch=2.0,
        log_history=[{"loss": 1.9, "epoch": 1.0}, {"loss": 1.4, "epoch": 2.0}],
    )
    callback.on_evaluate(None, state, None, metrics={"eval_loss": 1.1}, model="model")
    (record,) = evaluator.epochs
    assert (record["epoch"], record["train_loss"], record["eval_loss"]) == (2, 1.4, 1.1)


def test_the_callback_handles_a_log_history_without_a_training_loss(monkeypatch):
    _fake_unsloth(monkeypatch)
    _fake_transformers_callback(monkeypatch)
    evaluator = _evaluator()
    state = SimpleNamespace(epoch=1.0, log_history=[])
    train.make_epoch_eval_callback(evaluator).on_evaluate(
        None, state, None, metrics={"eval_loss": 1.1}, model="m"
    )
    assert evaluator.epochs[0]["train_loss"] is None


# ---- M8: the training record and rounds.json ----
def _trainer_state(**overrides):
    base = {
        "epoch": 3.0,
        "global_step": 72,
        "best_model_checkpoint": "/checkpoints/round_0/checkpoint-48",
        "best_metric": 0.91,
        "log_history": [{"loss": 1.5, "epoch": 1.0}],
    }
    return SimpleNamespace(**{**base, **overrides})


def _record(**overrides):
    base = {
        "round_n": 0,
        "trainer_state": _trainer_state(),
        "train_rows": [_trow(1), _trow(2), _trow(3)],
        "eval_rows": [_trow(4)],
        "requested_epochs": 3,
        "epoch_results": [{"epoch": 1}],
        "versions": {"python": "3.11"},
        "train_seconds": 90.5,
        "gpu_summary": {"gpu_name": None},
        "peak_torch_memory_allocated_mib": None,
        "hyperparams": train.hyperparameters(),
    }
    return train.build_training_record(**{**base, **overrides})


def test_training_record_has_the_fields_and_counts_match_the_rows():
    record = _record()
    for field in (
        "round",
        "model_name",
        "train_examples",
        "eval_examples",
        "requested_epochs",
        "epochs_run",
        "steps_run",
        "stopped_early",
        "best_model_checkpoint",
        "best_metric",
        "train_seconds",
        "versions",
        "gpu_summary",
        "peak_torch_memory_allocated_mib",
        "log_history",
        "epoch_results",
        "hyperparameters",
        "hyperparameter_hash",
        "gpu_type",
    ):
        assert field in record, field
    assert (record["train_examples"], record["eval_examples"]) == (3, 1)
    assert record["round"] == 0
    assert record["best_model_checkpoint"] == "/checkpoints/round_0/checkpoint-48"
    assert record["best_metric"] == 0.91
    assert json.loads(json.dumps(record)) == record


def test_training_record_says_when_training_stopped_early():
    assert _record(trainer_state=_trainer_state(epoch=2.0))["stopped_early"] is True
    assert _record(trainer_state=_trainer_state(epoch=3.0))["stopped_early"] is False


def test_training_record_hash_matches_the_hyperparameters():
    record = _record()
    assert record["hyperparameter_hash"] == train.hyperparameter_hash(
        record["hyperparameters"]
    )


def test_training_record_allows_no_best_checkpoint():
    record = _record(
        trainer_state=_trainer_state(best_model_checkpoint=None, best_metric=None)
    )
    assert (record["best_model_checkpoint"], record["best_metric"]) == (None, None)


def test_read_rounds_of_a_missing_file_is_empty(tmp_path):
    assert train.read_rounds(tmp_path / "rounds.json") == []


def test_read_rounds_of_a_corrupt_file_raises_naming_it(tmp_path):
    path = tmp_path / "rounds.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="rounds.json"):
        train.read_rounds(path)


def test_read_rounds_of_something_that_is_not_a_list_raises(tmp_path):
    path = tmp_path / "rounds.json"
    path.write_text('{"round": 0}', encoding="utf-8")
    with pytest.raises(ValueError, match="list"):
        train.read_rounds(path)


def test_record_round_appends_rounds_in_order(tmp_path):
    path = tmp_path / "results" / "rounds.json"
    train.record_round(path, {"round": 0, "x": 1})
    train.record_round(path, {"round": 1, "x": 2})
    assert [r["round"] for r in train.read_rounds(path)] == [0, 1]
    assert list(path.parent.iterdir()) == [path]


def test_record_round_refuses_a_round_that_is_already_recorded(tmp_path):
    path = tmp_path / "rounds.json"
    train.record_round(path, {"round": 0})
    with pytest.raises(ValueError, match="already"):
        train.record_round(path, {"round": 0})
    assert len(train.read_rounds(path)) == 1


@pytest.mark.parametrize(("existing", "new"), [([], 1), ([0], 2), ([0, 1], 3)])
def test_record_round_refuses_a_gap_in_the_round_numbers(tmp_path, existing, new):
    path = tmp_path / "rounds.json"
    for n in existing:
        train.record_round(path, {"round": n})
    with pytest.raises(ValueError, match="gap"):
        train.record_round(path, {"round": new})


# ---- M8: masking (the marker's tokenization and the labels Unsloth produces) ----
def _masked_example(tok, row, shift=0):
    full = tok.encode(
        data.format_example(row["question"], row["chain"], config.EOS_TOKEN),
        add_special_tokens=True,
    ).ids
    boundary = train.response_start_index(tok, row["question"]) + shift
    return {"input_ids": full, "labels": [-100] * boundary + full[boundary:]}


def test_response_start_index_is_the_length_of_the_prompt_including_the_start_token():
    tok = _tiny_tokenizer()
    assert train.response_start_index(tok, "a b") == 9


def test_expected_trained_text_is_the_response_after_the_marker_up_to_the_end_token():
    tok = _tiny_tokenizer()
    row = _trow(1, question="a b", chain=_teacher_chain("9"))
    text = train.expected_trained_text(tok, row)
    full = tok.encode(
        data.format_example(row["question"], row["chain"], config.EOS_TOKEN),
        add_special_tokens=True,
    ).ids
    assert text.rstrip().endswith(config.EOS_TOKEN)
    assert len(text.split()) == len(full) - train.response_start_index(tok, "a b")


def test_check_masking_accepts_labels_that_keep_only_the_response_and_end_token():
    tok = _tiny_tokenizer()
    rows = [_trow(1, question="a b"), _trow(2, question="c d e")]
    examples = [_masked_example(tok, r) for r in rows]
    assert train.check_masking(tok, rows, examples) is None


def test_check_masking_catches_an_off_by_one_that_leaves_the_marker_unmasked():
    tok = _tiny_tokenizer()
    row = _trow(1, question="a b")
    with pytest.raises(ValueError, match="t-1"):
        train.check_masking(tok, [row], [_masked_example(tok, row, shift=-1)])


def test_check_masking_catches_an_off_by_one_that_masks_the_first_response_token():
    tok = _tiny_tokenizer()
    row = _trow(1, question="a b")
    with pytest.raises(ValueError, match="t-1"):
        train.check_masking(tok, [row], [_masked_example(tok, row, shift=1)])


def test_check_masking_catches_a_masked_end_token():
    tok = _tiny_tokenizer()
    row = _trow(1, question="a b")
    example = _masked_example(tok, row)
    example["labels"][-1] = -100
    with pytest.raises(ValueError, match="t-1"):
        train.check_masking(tok, [row], [example])


def test_check_masking_catches_everything_masked():
    tok = _tiny_tokenizer()
    row = _trow(1, question="a b")
    example = _masked_example(tok, row)
    example["labels"] = [-100] * len(example["labels"])
    with pytest.raises(ValueError, match="t-1"):
        train.check_masking(tok, [row], [example])


def test_check_masking_with_different_counts_of_rows_and_examples_raises():
    tok = _tiny_tokenizer()
    with pytest.raises(ValueError, match="1 rows but 0"):
        train.check_masking(tok, [_trow(1)], [])


@pytest.mark.slow
def test_real_tokenizer_marker_boundary_has_no_off_by_one():
    tok = data.load_tokenizer(config.MODEL_NAME)
    for question, chain in (
        (LONG_Q, _teacher_chain("9")),
        ("Was Talos alive?", _teacher_chain("No", "Talos was a bronze automaton.")),
    ):
        row = _trow(1, question=question, chain=chain)
        assert train.expected_trained_text(tok, row) == chain + config.EOS_TOKEN
        prompt_ids = tok.encode(
            data.format_prompt(question), add_special_tokens=True
        ).ids
        assert tok.decode(prompt_ids[-4:], skip_special_tokens=False).endswith(
            config.RESPONSE_MARKER
        )


@pytest.mark.slow
def test_real_tokenizer_masked_labels_keep_exactly_the_response_and_end_token():
    tok = data.load_tokenizer(config.MODEL_NAME)
    rows = [
        _trow(1, question=LONG_Q, chain=_teacher_chain("9")),
        _trow(
            2,
            question="How do planes stay in the air?",
            chain=_teacher_chain("9", "wings"),
        ),
    ]
    examples = [_masked_example(tok, r) for r in rows]
    assert train.check_masking(tok, rows, examples) is None


def test_masking_helpers_work_with_a_transformers_style_tokenizer():
    inner = _tiny_tokenizer()
    hf = _HFStyleTokenizer(inner)
    row = _trow(1, question="a b")
    assert train.response_start_index(hf, "a b") == train.response_start_index(
        inner, "a b"
    )
    assert train.expected_trained_text(hf, row) == train.expected_trained_text(
        inner, row
    )
    example = _masked_example(inner, row)
    assert train.check_masking(hf, [row], [example]) is None
    shifted = _masked_example(inner, row, shift=-1)
    with pytest.raises(ValueError, match="t-1"):
        train.check_masking(hf, [row], [shifted])


# ---- M8: masking on the collated batch (padding-free packs examples into one row) ----
def _example_ids_and_labels(tok, row, shift=0):
    example = _masked_example(tok, row, shift=shift)
    return example["input_ids"], example["labels"]


def _packed_batch(tok, rows, shifts=None):
    shifts = shifts or [0] * len(rows)
    ids, labels = [], []
    for row, shift in zip(rows, shifts, strict=True):
        row_ids, row_labels = _example_ids_and_labels(tok, row, shift)
        ids += row_ids
        labels += row_labels
    return {"input_ids": [ids], "labels": [labels]}


def _padded_batch(tok, rows):
    pairs = [_example_ids_and_labels(tok, row) for row in rows]
    width = max(len(ids) for ids, _ in pairs)
    return {
        "input_ids": [ids + [0] * (width - len(ids)) for ids, _ in pairs],
        "labels": [labels + [-100] * (width - len(labels)) for _, labels in pairs],
    }


def _two_rows():
    return [
        _trow(1, question="a b", chain=_teacher_chain("9")),
        _trow(2, question="c d e f", chain=_teacher_chain("9", "extra words here")),
    ]


def test_unmasked_runs_splits_contiguous_trained_stretches():
    assert train.unmasked_runs([1, 2, 3, 4, 5, 6], [-100, 2, 3, -100, 5, 6]) == [
        [2, 3],
        [5, 6],
    ]
    assert train.unmasked_runs([1, 2], [-100, -100]) == []
    assert train.unmasked_runs([1, 2], [1, 2]) == [[1, 2]]


def test_check_batch_masking_accepts_a_packed_padding_free_batch():
    tok = _tiny_tokenizer()
    rows = _two_rows()
    assert (
        train.check_batch_masking(tok, rows, _packed_batch(tok, rows), expected_count=2)
        is None
    )


def test_check_batch_masking_accepts_a_padded_batch():
    tok = _tiny_tokenizer()
    rows = _two_rows()
    assert (
        train.check_batch_masking(tok, rows, _padded_batch(tok, rows), expected_count=2)
        is None
    )


def test_check_batch_masking_accepts_tensor_like_batches():
    tok = _tiny_tokenizer()
    rows = _two_rows()
    batch = {key: np.array(value) for key, value in _padded_batch(tok, rows).items()}
    assert train.check_batch_masking(tok, rows, batch) is None


def test_check_batch_masking_accepts_the_examples_in_either_order():
    tok = _tiny_tokenizer()
    rows = _two_rows()
    assert train.check_batch_masking(tok, rows, _packed_batch(tok, rows[::-1])) is None


def test_check_batch_masking_catches_a_boundary_that_merges_two_examples():
    tok = _tiny_tokenizer()
    rows = _two_rows()
    batch = _packed_batch(tok, rows)
    first_len = len(_example_ids_and_labels(tok, rows[0])[0])
    start = train.response_start_index(tok, rows[1]["question"])
    for i in range(
        first_len, first_len + start
    ):  # the second example's prompt left unmasked
        batch["labels"][0][i] = batch["input_ids"][0][i]
    with pytest.raises(ValueError, match="does not match"):
        train.check_batch_masking(tok, rows, batch)


@pytest.mark.parametrize("shift", [-1, 1])
def test_check_batch_masking_catches_an_off_by_one_in_a_packed_batch(shift):
    tok = _tiny_tokenizer()
    rows = _two_rows()
    with pytest.raises(ValueError, match="does not match"):
        train.check_batch_masking(
            tok, rows, _packed_batch(tok, rows, shifts=[0, shift])
        )


def test_check_batch_masking_catches_a_masked_end_token():
    tok = _tiny_tokenizer()
    rows = _two_rows()
    batch = _packed_batch(tok, rows)
    batch["labels"][0][-1] = -100
    with pytest.raises(ValueError, match="does not match"):
        train.check_batch_masking(tok, rows, batch)


def test_check_batch_masking_with_nothing_trained_raises():
    tok = _tiny_tokenizer()
    rows = _two_rows()
    batch = _packed_batch(tok, rows)
    batch["labels"] = [[-100] * len(batch["labels"][0])]
    with pytest.raises(ValueError, match="no trained tokens"):
        train.check_batch_masking(tok, rows, batch)


def test_check_batch_masking_checks_the_number_of_examples_when_it_is_known():
    tok = _tiny_tokenizer()
    rows = _two_rows()
    with pytest.raises(ValueError, match="1 trained"):
        train.check_batch_masking(
            tok, rows, _packed_batch(tok, rows[:1]), expected_count=2
        )


def test_check_batch_masking_works_with_a_transformers_style_tokenizer():
    inner = _tiny_tokenizer()
    rows = _two_rows()
    assert (
        train.check_batch_masking(
            _HFStyleTokenizer(inner), rows, _packed_batch(inner, rows)
        )
        is None
    )


@pytest.mark.slow
def test_real_tokenizer_packed_batch_masking_keeps_exactly_the_responses():
    tok = data.load_tokenizer(config.MODEL_NAME)
    rows = [
        _trow(1, question=LONG_Q, chain=_teacher_chain("9")),
        _trow(
            2,
            question="How do planes stay in the air?",
            chain=_teacher_chain("9", "wings"),
        ),
        _trow(
            3,
            question="What makes bread rise in the oven?",
            chain=_teacher_chain("9", "yeast"),
        ),
    ]
    assert (
        train.check_batch_masking(tok, rows, _packed_batch(tok, rows), expected_count=3)
        is None
    )


# ---- M9: plan_round, round_decision, resume loading ----
def _recorded(n, settings=None):
    settings = settings or train.hyperparameters()
    return [
        {"round": i, "hyperparameter_hash": train.hyperparameter_hash(settings)}
        for i in range(n)
    ]


def test_round_zero_starts_from_the_base_model():
    assert train.plan_round(0, [], train.hyperparameters()) is None


def test_round_zero_refuses_a_resume_adapter():
    with pytest.raises(ValueError, match="base model"):
        train.plan_round(
            0, [], train.hyperparameters(), resume_adapter="/checkpoints/round_0"
        )


def test_a_later_round_resumes_from_the_previous_rounds_adapter():
    path = train.plan_round(1, _recorded(1), train.hyperparameters())
    assert path == "/checkpoints/round_0"
    assert (
        train.plan_round(3, _recorded(3), train.hyperparameters())
        == "/checkpoints/round_2"
    )


def test_the_given_resume_adapter_must_match_the_expected_one():
    ok = train.plan_round(
        1, _recorded(1), train.hyperparameters(), resume_adapter="/checkpoints/round_0"
    )
    assert ok == "/checkpoints/round_0"
    with pytest.raises(ValueError, match="wrong resume adapter"):
        train.plan_round(
            2,
            _recorded(2),
            train.hyperparameters(),
            resume_adapter="/checkpoints/round_0",
        )


def test_a_round_beyond_the_maximum_is_refused():
    beyond = config.MAX_ROUNDS + 1
    with pytest.raises(ValueError, match="maximum"):
        train.plan_round(beyond, _recorded(beyond), train.hyperparameters())
    assert train.plan_round(
        config.MAX_ROUNDS, _recorded(config.MAX_ROUNDS), train.hyperparameters()
    )


def test_a_round_that_is_already_recorded_is_refused():
    with pytest.raises(ValueError, match="already"):
        train.plan_round(1, _recorded(2), train.hyperparameters())


@pytest.mark.parametrize(("recorded", "round_n"), [(0, 1), (1, 2), (2, 3)])
def test_a_round_that_skips_one_is_refused(recorded, round_n):
    with pytest.raises(ValueError, match="gap"):
        train.plan_round(round_n, _recorded(recorded), train.hyperparameters())


def test_changed_hyperparameters_are_refused_by_comparing_the_recorded_hash():
    rounds = _recorded(1, train.hyperparameters(learning_rate=1e-4))
    with pytest.raises(ValueError, match="hyperparameters changed"):
        train.plan_round(1, rounds, train.hyperparameters())


def test_round_decision_stops_when_nothing_is_weak_and_continues_otherwise():
    assert train.round_decision([]) == "stop"
    assert train.round_decision(["pun_ambiguity"]) == "continue"


def test_load_model_for_training_can_resume_from_an_adapter(monkeypatch):
    fast = _fake_unsloth(monkeypatch)
    train.load_model_for_training(resume_adapter="/checkpoints/round_0")
    assert fast.from_pretrained.call_args.kwargs["model_name"] == "/checkpoints/round_0"
    assert fast.from_pretrained.call_args.kwargs["load_in_4bit"] is True
    fast.get_peft_model.assert_called_once()


# ---- M9: the epoch whose adapter is kept ----
def test_best_epoch_is_the_one_with_the_lowest_eval_loss():
    epochs = [
        {"epoch": 1, "eval_loss": 1.2},
        {"epoch": 2, "eval_loss": 0.9},
        {"epoch": 3, "eval_loss": 1.0},
    ]
    assert train.best_epoch(epochs)["epoch"] == 2


def test_best_epoch_ties_go_to_the_earlier_epoch():
    epochs = [{"epoch": 1, "eval_loss": 0.9}, {"epoch": 2, "eval_loss": 0.9}]
    assert train.best_epoch(epochs)["epoch"] == 1


def test_best_epoch_ignores_epochs_without_an_eval_loss():
    epochs = [{"epoch": 1, "eval_loss": None}, {"epoch": 2, "eval_loss": 1.1}]
    assert train.best_epoch(epochs)["epoch"] == 2


def test_best_epoch_with_no_losses_falls_back_to_the_last_epoch():
    epochs = [{"epoch": 1, "eval_loss": None}, {"epoch": 2, "eval_loss": None}]
    assert train.best_epoch(epochs)["epoch"] == 2


def test_best_epoch_of_nothing_raises():
    with pytest.raises(ValueError, match="no epochs"):
        train.best_epoch([])
