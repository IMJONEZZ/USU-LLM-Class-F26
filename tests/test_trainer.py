import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from src.trainer import (
    GIB,
    LORA_TARGET_MODULES,
    TrainingConfig,
    TrainingResult,
    build_model_and_tokenizer,
    build_trainer,
    evaluation_interval,
    format_example,
    main,
    run_training,
    stratified_partition,
    stratified_sample,
    train_sst2,
    write_result,
)


def labeled_examples(per_label=20):
    return [
        {
            "id": f"{label}-{index}",
            "sentence": f"review {label}-{index}",
            "label": label,
        }
        for label in (0, 1)
        for index in range(per_label)
    ]


def test_training_config_defaults_match_action_plan():
    config = TrainingConfig()

    assert config.model_id == "meta-llama/Llama-3.2-1B-Instruct"
    assert config.effective_batch_size == 16
    assert config.learning_rate == 2e-4
    assert config.epochs == 2
    assert config.max_sequence_length == 128
    assert config.lora_rank == config.lora_alpha == 16
    assert config.seed == 42


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("batch_size", 0, "batch_size"),
        ("development_fraction", 0.0, "development_fraction"),
        ("learning_rate", 0.0, "learning_rate"),
        ("lora_dropout", 1.0, "lora_dropout"),
        ("weight_decay", -0.1, "weight_decay"),
        ("warmup_ratio", 1.0, "warmup_ratio"),
    ],
)
def test_training_config_rejects_invalid_values(field, value, message):
    with pytest.raises(ValueError, match=message):
        TrainingConfig(**{field: value})


@pytest.mark.parametrize(
    ("label", "word"),
    [(0, "negative"), (1, "positive")],
)
def test_format_example_builds_prompt_completion(label, word):
    formatted = format_example({"sentence": "A movie.", "label": label})

    assert formatted["prompt"][-1]["content"] == "A movie."
    assert formatted["completion"] == [{"role": "assistant", "content": word}]


@pytest.mark.parametrize(
    "example",
    [
        {"sentence": "", "label": 0},
        {"sentence": "review", "label": 2},
    ],
)
def test_format_example_rejects_invalid_rows(example):
    with pytest.raises(ValueError):
        format_example(example)


def test_stratified_partition_is_deterministic_balanced_and_disjoint():
    examples = labeled_examples()

    first_train, first_development = stratified_partition(
        examples, development_fraction=0.10, seed=42
    )
    second_train, second_development = stratified_partition(
        examples, development_fraction=0.10, seed=42
    )

    assert first_train == second_train
    assert first_development == second_development
    assert len(first_train) == 36
    assert len(first_development) == 4
    assert {row["label"] for row in first_development} == {0, 1}
    assert {row["id"] for row in first_train}.isdisjoint(
        row["id"] for row in first_development
    )


def test_stratified_sample_is_exact_balanced_and_from_training_pool():
    training, development = stratified_partition(labeled_examples(per_label=50))

    sample = stratified_sample(training, 20, seed=42)

    assert len(sample) == 20
    assert sum(row["label"] == 0 for row in sample) == 10
    assert {row["id"] for row in sample} <= {row["id"] for row in training}
    assert {row["id"] for row in sample}.isdisjoint(row["id"] for row in development)
    assert sample == stratified_sample(training, 20, seed=42)


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda rows: stratified_partition(rows, development_fraction=0), "fraction"),
        (lambda rows: stratified_sample(rows, 0), "positive"),
        (lambda rows: stratified_sample(rows, len(rows) + 1), "exceed"),
        (lambda rows: stratified_partition(rows[:1]), "both"),
    ],
)
def test_split_helpers_reject_invalid_inputs(operation, message):
    with pytest.raises(ValueError, match=message):
        operation(labeled_examples(per_label=2))


def test_build_model_uses_non_quantized_lora_configuration():
    calls = {}

    class FakeFastLanguageModel:
        @staticmethod
        def from_pretrained(**kwargs):
            calls["load"] = kwargs
            return "base-model", "tokenizer"

        @staticmethod
        def get_peft_model(model, **kwargs):
            calls["lora"] = {"model": model, **kwargs}
            return "lora-model"

    model, tokenizer = build_model_and_tokenizer(
        TrainingConfig(), FakeFastLanguageModel
    )

    assert (model, tokenizer) == ("lora-model", "tokenizer")
    assert calls["load"]["load_in_4bit"] is False
    assert calls["load"]["max_seq_length"] == 128
    assert calls["lora"]["r"] == 16
    assert calls["lora"]["target_modules"] == list(LORA_TARGET_MODULES)
    assert calls["lora"]["use_gradient_checkpointing"] == "unsloth"


def test_evaluation_interval_uses_planned_limit_and_pilot_epoch():
    config = TrainingConfig(evaluation_steps=1_000)

    assert evaluation_interval(2_000, config) == 125
    assert evaluation_interval(64_000, config) == 1_000
    with pytest.raises(ValueError, match="training_size"):
        evaluation_interval(0, config)


def test_build_trainer_passes_planned_arguments():
    captured = {}

    def fake_config(**kwargs):
        captured["arguments"] = kwargs
        return "training-arguments"

    def fake_trainer(**kwargs):
        captured["trainer"] = kwargs
        return "trainer"

    trainer = build_trainer(
        "model",
        "tokenizer",
        list(range(2_000)),
        ["development"],
        TrainingConfig(),
        epoch_end_callback="epoch-callback",
        sft_config_factory=fake_config,
        sft_trainer_factory=fake_trainer,
    )

    assert trainer == "trainer"
    assert captured["arguments"]["eval_steps"] == 125
    assert captured["arguments"]["completion_only_loss"] is True
    assert captured["arguments"]["fp16"] is True
    assert captured["arguments"]["data_seed"] == 42
    assert captured["trainer"]["processing_class"] == "tokenizer"
    assert captured["trainer"]["callbacks"] == ["epoch-callback"]


class FakeCuda:
    def __init__(self):
        self.reset_calls = 0
        self.synchronize_calls = 0

    def reset_peak_memory_stats(self):
        self.reset_calls += 1

    def synchronize(self):
        self.synchronize_calls += 1

    @staticmethod
    def max_memory_allocated():
        return 3 * GIB

    @staticmethod
    def max_memory_reserved():
        return 4 * GIB


class FakeTrainer:
    def __init__(self):
        self.state = SimpleNamespace(
            log_history=[
                {"loss": 1.2, "step": 1},
                {"eval_loss": 0.9, "step": 1},
                {"loss": 0.4, "step": 2},
                {"eval_loss": 0.3, "step": 2},
            ]
        )
        self.saved_to = None

    @staticmethod
    def train():
        return SimpleNamespace(metrics={"train_runtime": 7.5})

    def save_model(self, path):
        self.saved_to = path


class FakeTokenizer:
    def __init__(self):
        self.saved_to = None

    def save_pretrained(self, path):
        self.saved_to = path


def test_run_training_collects_losses_time_memory_and_saves(tmp_path):
    trainer = FakeTrainer()
    tokenizer = FakeTokenizer()
    cuda = FakeCuda()
    times = iter([10.0, 18.5])
    config = TrainingConfig(output_dir=str(tmp_path / "adapter"))

    result = run_training(
        trainer,
        tokenizer,
        config,
        pilot=True,
        training_examples=2_000,
        development_examples=100,
        torch_module=SimpleNamespace(cuda=cuda),
        clock=lambda: next(times),
    )

    assert result.wall_time_seconds == 8.5
    assert result.first_training_loss == 1.2
    assert result.final_training_loss == 0.4
    assert result.first_development_loss == 0.9
    assert result.final_development_loss == 0.3
    assert result.peak_memory_allocated_gib == 3.0
    assert result.peak_memory_reserved_gib == 4.0
    assert cuda.reset_calls == 1
    assert cuda.synchronize_calls == 2
    assert trainer.saved_to == tokenizer.saved_to == str(tmp_path / "adapter")


def test_write_result_creates_json_parent_directories(tmp_path):
    result = TrainingResult(
        pilot=True,
        training_examples=2,
        development_examples=1,
        wall_time_seconds=1.0,
        trainer_metrics={},
        peak_memory_allocated_gib=1.0,
        peak_memory_reserved_gib=2.0,
        first_training_loss=None,
        final_training_loss=None,
        first_development_loss=None,
        final_development_loss=None,
        log_history=(),
        config={},
    )
    destination = tmp_path / "nested" / "result.json"

    write_result(result, destination)

    assert json.loads(destination.read_text())["pilot"] is True


def test_train_sst2_orchestrates_pilot_without_leakage(tmp_path):
    raw = labeled_examples(per_label=20)

    class FakeDataset(list):
        def __getitem__(self, key):
            if isinstance(key, str):
                return [row[key] for row in self]
            return super().__getitem__(key)

        @classmethod
        def from_list(cls, rows):
            return cls(rows)

    class FakeFastLanguageModel:
        @staticmethod
        def from_pretrained(**kwargs):
            return "model", FakeTokenizer()

        @staticmethod
        def get_peft_model(model, **kwargs):
            return model

    def fake_config(**kwargs):
        return kwargs

    trainer = FakeTrainer()

    def fake_trainer(**kwargs):
        trainer.datasets = (kwargs["train_dataset"], kwargs["eval_dataset"])
        return trainer

    stack = SimpleNamespace(
        dataset_class=FakeDataset,
        load_dataset=lambda *args, **kwargs: FakeDataset(raw),
        fast_language_model=FakeFastLanguageModel,
        epoch_end_callback="epoch-callback",
        sft_config=fake_config,
        sft_trainer=fake_trainer,
        torch=SimpleNamespace(cuda=FakeCuda(), inference_mode=nullcontext),
        train_runner=lambda selected_trainer: selected_trainer.train(),
    )
    config = TrainingConfig(
        pilot_size=10,
        development_fraction=0.10,
        output_dir=str(tmp_path / "adapter"),
        result_path=str(tmp_path / "result.json"),
    )

    result = train_sst2(config, pilot=True, stack_loader=lambda: stack)

    assert result.training_examples == 10
    assert result.development_examples == 4
    assert len(trainer.datasets[0]) == 10
    assert len(trainer.datasets[1]) == 4
    assert (tmp_path / "result.json").exists()


def test_main_prints_training_result(monkeypatch, capsys, tmp_path):
    result = TrainingResult(
        pilot=True,
        training_examples=2,
        development_examples=1,
        wall_time_seconds=1.0,
        trainer_metrics={},
        peak_memory_allocated_gib=1.0,
        peak_memory_reserved_gib=2.0,
        first_training_loss=None,
        final_training_loss=None,
        first_development_loss=None,
        final_development_loss=None,
        log_history=(),
        config={},
    )
    received = {}

    def fake_train(config, *, pilot):
        received["config"] = config
        received["pilot"] = pilot
        return result

    monkeypatch.setattr("src.trainer.train_sst2", fake_train)
    main(
        [
            "--pilot",
            "--output-dir",
            str(tmp_path / "adapter"),
            "--result-path",
            str(tmp_path / "result.json"),
        ]
    )

    assert received["pilot"] is True
    assert received["config"].output_dir == str(tmp_path / "adapter")
    assert json.loads(capsys.readouterr().out)["pilot"] is True
