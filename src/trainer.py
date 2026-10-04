"""LoRA fine-tuning pipeline for Llama 3.2 1B Instruct on SST-2.

GPU libraries are imported only by :func:`load_training_stack`, keeping this
module importable in the repository's CPU-only CI environment.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.evaluator import (
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    LABEL_NAMES,
    sentiment_messages,
)

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
GIB = 1024**3


@dataclass(frozen=True)
class TrainingConfig:
    """All reproducible settings for the pilot and final training runs."""

    model_id: str = MODEL_ID
    seed: int = 42
    development_fraction: float = 0.05
    pilot_size: int = 2_000
    learning_rate: float = 2e-4
    epochs: int = 2
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_sequence_length: int = 128
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    scheduler: str = "linear"
    logging_steps: int = 100
    evaluation_steps: int = 1_000
    output_dir: str = "assignment_5/artifacts/adapter"
    result_path: str = "assignment_5/artifacts/training_result.json"

    def __post_init__(self) -> None:
        positive_integers = {
            "seed": self.seed,
            "pilot_size": self.pilot_size,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_sequence_length": self.max_sequence_length,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "logging_steps": self.logging_steps,
            "evaluation_steps": self.evaluation_steps,
        }
        for name, value in positive_integers.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.development_fraction < 1.0:
            raise ValueError("development_fraction must be between 0 and 1")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")

    @property
    def effective_batch_size(self) -> int:
        """Number of examples contributing to one optimizer update."""

        return self.batch_size * self.gradient_accumulation_steps


@dataclass(frozen=True)
class TrainingStack:
    """Lazy-loaded external dependencies used only in the GPU environment."""

    dataset_class: Any
    load_dataset: Callable[..., Any]
    fast_language_model: Any
    epoch_end_callback: Any
    sft_config: Any
    sft_trainer: Any
    torch: Any
    train_runner: Callable[[Any], Any]


@dataclass(frozen=True)
class TrainingResult:
    """Serializable training measurements and loss history."""

    pilot: bool
    training_examples: int
    development_examples: int
    wall_time_seconds: float
    trainer_metrics: dict[str, Any]
    peak_memory_allocated_gib: float
    peak_memory_reserved_gib: float
    first_training_loss: float | None
    final_training_loss: float | None
    first_development_loss: float | None
    final_development_loss: float | None
    log_history: tuple[dict[str, Any], ...]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert nested dataclasses and tuples for JSON serialization."""

        return asdict(self)


def format_example(example: Mapping[str, Any]) -> dict[str, Any]:
    """Convert an SST-2 row into conversational prompt-completion format."""

    label = int(example["label"])
    if label not in LABEL_NAMES:
        raise ValueError(f"label must be 0 or 1; got {label}")
    sentence = str(example["sentence"])
    if not sentence.strip():
        raise ValueError("sentence cannot be empty")
    return {
        "prompt": sentiment_messages(sentence),
        "completion": [{"role": "assistant", "content": LABEL_NAMES[label]}],
    }


def _group_by_label(
    examples: Sequence[Mapping[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        label = int(example["label"])
        if label not in LABEL_NAMES:
            raise ValueError(f"label must be 0 or 1; got {label}")
        grouped[label].append(dict(example))
    if set(grouped) != set(LABEL_NAMES):
        raise ValueError("examples must contain both sentiment labels")
    return grouped


def stratified_partition(
    examples: Sequence[Mapping[str, Any]],
    *,
    development_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create one deterministic label-stratified train/development partition."""

    if not 0.0 < development_fraction < 1.0:
        raise ValueError("development_fraction must be between 0 and 1")
    grouped = _group_by_label(examples)
    rng = random.Random(seed)
    training: list[dict[str, Any]] = []
    development: list[dict[str, Any]] = []
    for label in sorted(grouped):
        rows = grouped[label]
        rng.shuffle(rows)
        development_count = max(1, round(len(rows) * development_fraction))
        if development_count >= len(rows):
            raise ValueError("each label needs at least one training example")
        development.extend(rows[:development_count])
        training.extend(rows[development_count:])
    rng.shuffle(training)
    rng.shuffle(development)
    return training, development


def stratified_sample(
    examples: Sequence[Mapping[str, Any]],
    sample_size: int,
    *,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Select an exact-size deterministic sample while preserving label balance."""

    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if sample_size > len(examples):
        raise ValueError("sample_size cannot exceed the available examples")
    grouped = _group_by_label(examples)
    exact_quotas = {
        label: sample_size * len(rows) / len(examples)
        for label, rows in grouped.items()
    }
    quotas = {label: math.floor(value) for label, value in exact_quotas.items()}
    remaining = sample_size - sum(quotas.values())
    remainders = sorted(
        grouped,
        key=lambda label: (exact_quotas[label] - quotas[label], -label),
        reverse=True,
    )
    for label in remainders[:remaining]:
        quotas[label] += 1

    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    for label in sorted(grouped):
        rows = grouped[label]
        rng.shuffle(rows)
        sample.extend(rows[: quotas[label]])
    rng.shuffle(sample)
    return sample


def load_training_stack() -> TrainingStack:
    """Import GPU and network dependencies only when a real run begins."""

    import torch
    from datasets import Dataset, load_dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel, unsloth_train

    class EvaluateAtEpochEnd(TrainerCallback):
        """Request evaluation and a checkpoint at every epoch boundary."""

        def on_epoch_end(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            control.should_evaluate = True
            control.should_save = True
            return control

    return TrainingStack(
        dataset_class=Dataset,
        load_dataset=load_dataset,
        fast_language_model=FastLanguageModel,
        epoch_end_callback=EvaluateAtEpochEnd(),
        sft_config=SFTConfig,
        sft_trainer=SFTTrainer,
        torch=torch,
        train_runner=unsloth_train,
    )


def build_model_and_tokenizer(
    config: TrainingConfig,
    fast_language_model: Any,
) -> tuple[Any, Any]:
    """Load the non-quantized base model and attach LoRA adapters."""

    model, tokenizer = fast_language_model.from_pretrained(
        model_name=config.model_id,
        max_seq_length=config.max_sequence_length,
        dtype=None,
        load_in_4bit=False,
    )
    model = fast_language_model.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=list(LORA_TARGET_MODULES),
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
        use_rslora=False,
        loftq_config=None,
    )
    return model, tokenizer


def evaluation_interval(training_size: int, config: TrainingConfig) -> int:
    """Evaluate every planned interval and at least once per pilot epoch."""

    if training_size < 1:
        raise ValueError("training_size must be positive")
    optimizer_steps_per_epoch = math.ceil(training_size / config.effective_batch_size)
    return min(config.evaluation_steps, optimizer_steps_per_epoch)


def build_trainer(
    model: Any,
    tokenizer: Any,
    training_dataset: Any,
    development_dataset: Any,
    config: TrainingConfig,
    *,
    epoch_end_callback: Any,
    sft_config_factory: Callable[..., Any],
    sft_trainer_factory: Callable[..., Any],
) -> Any:
    """Create TRL's supervised fine-tuning trainer with planned settings."""

    interval = evaluation_interval(len(training_dataset), config)
    arguments = sft_config_factory(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_length=config.max_sequence_length,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=interval,
        save_strategy="steps",
        save_steps=interval,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.scheduler,
        fp16=True,
        bf16=False,
        seed=config.seed,
        data_seed=config.seed,
        report_to="none",
        completion_only_loss=True,
    )
    return sft_trainer_factory(
        model=model,
        processing_class=tokenizer,
        train_dataset=training_dataset,
        eval_dataset=development_dataset,
        args=arguments,
        callbacks=[epoch_end_callback],
    )


def _loss_bounds(
    history: Sequence[Mapping[str, Any]], key: str
) -> tuple[float | None, float | None]:
    values = [float(item[key]) for item in history if key in item]
    if not values:
        return None, None
    return values[0], values[-1]


def run_training(
    trainer: Any,
    tokenizer: Any,
    config: TrainingConfig,
    *,
    pilot: bool,
    training_examples: int,
    development_examples: int,
    torch_module: Any,
    train_runner: Callable[[Any], Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> TrainingResult:
    """Run training while collecting wall-time, loss, and peak-VRAM evidence."""

    torch_module.cuda.reset_peak_memory_stats()
    torch_module.cuda.synchronize()
    started = clock()
    trainer_output = trainer.train() if train_runner is None else train_runner(trainer)
    torch_module.cuda.synchronize()
    wall_time = clock() - started

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    history = tuple(dict(item) for item in trainer.state.log_history)
    first_training_loss, final_training_loss = _loss_bounds(history, "loss")
    first_development_loss, final_development_loss = _loss_bounds(history, "eval_loss")
    return TrainingResult(
        pilot=pilot,
        training_examples=training_examples,
        development_examples=development_examples,
        wall_time_seconds=wall_time,
        trainer_metrics=dict(trainer_output.metrics),
        peak_memory_allocated_gib=(torch_module.cuda.max_memory_allocated() / GIB),
        peak_memory_reserved_gib=torch_module.cuda.max_memory_reserved() / GIB,
        first_training_loss=first_training_loss,
        final_training_loss=final_training_loss,
        first_development_loss=first_development_loss,
        final_development_loss=final_development_loss,
        log_history=history,
        config=asdict(config),
    )


def write_result(result: TrainingResult, path: str | Path) -> None:
    """Write one reproducible JSON evidence file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def train_sst2(
    config: TrainingConfig,
    *,
    pilot: bool,
    stack_loader: Callable[[], TrainingStack] = load_training_stack,
) -> TrainingResult:
    """Load SST-2, create leak-free splits, and run one pilot or final job."""

    stack = stack_loader()
    dataset = stack.load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split="train",
        revision=DATASET_REVISION,
    )
    examples = [
        {"sentence": sentence, "label": int(label)}
        for sentence, label in zip(dataset["sentence"], dataset["label"], strict=True)
    ]
    training_pool, development = stratified_partition(
        examples,
        development_fraction=config.development_fraction,
        seed=config.seed,
    )
    training = (
        stratified_sample(training_pool, config.pilot_size, seed=config.seed)
        if pilot
        else training_pool
    )
    formatted_training = stack.dataset_class.from_list(
        [format_example(example) for example in training]
    )
    formatted_development = stack.dataset_class.from_list(
        [format_example(example) for example in development]
    )
    model, tokenizer = build_model_and_tokenizer(config, stack.fast_language_model)
    trainer = build_trainer(
        model,
        tokenizer,
        formatted_training,
        formatted_development,
        config,
        epoch_end_callback=stack.epoch_end_callback,
        sft_config_factory=stack.sft_config,
        sft_trainer_factory=stack.sft_trainer,
    )
    result = run_training(
        trainer,
        tokenizer,
        config,
        pilot=pilot,
        training_examples=len(training),
        development_examples=len(development),
        torch_module=stack.torch,
        train_runner=stack.train_runner,
    )
    write_result(result, config.result_path)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    """Run the network-backed trainer and print its JSON result."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--output-dir", default=TrainingConfig.output_dir)
    parser.add_argument("--result-path", default=TrainingConfig.result_path)
    args = parser.parse_args(argv)
    config = TrainingConfig(
        output_dir=args.output_dir,
        result_path=args.result_path,
    )
    result = train_sst2(config, pilot=args.pilot)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - manual GPU entry point
    main()
