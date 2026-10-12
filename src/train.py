import hashlib
import json
from pathlib import Path

from src import config, data
from src import eval as evaluation

# Llama 3.2 1B shape (public model card values); every figure below is a rough
# ESTIMATE, not a measurement: the two-step probe in M8 measures the real peak.
_NON_EMBEDDING_PARAMS = 973_000_000
_EMBEDDING_PARAMS = 262_668_288
_HIDDEN = 2048
_LAYERS = 16
_VOCAB = 128_256
_LORA_PARAMS_PER_RANK = 44_032 * _LAYERS  # q,k,v,o,gate,up,down in every layer
_ONE_LAYER_WORKING_BYTES_PER_TOKEN = 81_920
_OVERHEAD_GIB = 1.5  # CUDA context and allocator fragmentation
_GIB = 1024**3


def estimate_training_memory_gib(max_tokens, batch_size=1):
    if max_tokens <= 0 or batch_size <= 0:
        raise ValueError("max_tokens and batch_size must be positive")
    tokens = max_tokens * batch_size
    parts = {
        # 4-bit weights, with the embeddings kept in 16 bits
        "weights": (_NON_EMBEDDING_PARAMS * 0.5 + _EMBEDDING_PARAMS * 2) / _GIB,
        # LoRA weights, gradients and Adam state in 32 bits: 16 bytes per parameter
        "lora_and_optimizer": config.LORA_R * _LORA_PARAMS_PER_RANK * 16 / _GIB,
        # gradient checkpointing: one saved input per layer plus one layer's workspace
        "activations": tokens
        * (_HIDDEN * 2 * _LAYERS + _ONE_LAYER_WORKING_BYTES_PER_TOKEN)
        / _GIB,
        # float32 logits, their gradient and the softmax copy
        "logits": tokens * _VOCAB * 4 * 3 / _GIB,
        "overhead": _OVERHEAD_GIB,
    }
    parts["total"] = sum(parts.values())
    return parts


def choose_gpu(required_gib, options=None, usable_fraction=None):
    options = config.GPU_OPTIONS if options is None else options
    fraction = (
        config.GPU_USABLE_FRACTION if usable_fraction is None else usable_fraction
    )
    fitting = [
        (option["usd_per_second"], option["vram_gib"], name)
        for name, option in options.items()
        if option["vram_gib"] * fraction >= required_gib
    ]
    if not fitting:
        raise ValueError(f"no GPU fits {required_gib:.1f} GiB")
    return min(fitting)[2]


def steps_per_epoch(n_examples, per_device_batch, grad_accum):
    if n_examples <= 0 or per_device_batch <= 0 or grad_accum <= 0:
        raise ValueError("examples, batch size and accumulation must be positive")
    batches = -(-n_examples // per_device_batch)
    return -(-batches // grad_accum)


def total_steps(n_examples, per_device_batch, grad_accum, epochs):
    return steps_per_epoch(n_examples, per_device_batch, grad_accum) * epochs


def clean_overrides(overrides):
    return {name: value for name, value in overrides.items() if value is not None}


def hyperparameters(**overrides):
    base = {
        "model": config.MODEL_NAME,
        "learning_rate": config.LEARNING_RATE,
        "epochs": config.EPOCHS,
        "per_device_train_batch_size": config.PER_DEVICE_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": config.GRADIENT_ACCUMULATION_STEPS,
        "lora_r": config.LORA_R,
        "lora_alpha": config.LORA_ALPHA,
        "lora_target_modules": list(config.LORA_TARGET_MODULES),
        "max_seq_length": config.MAX_SEQ_LENGTH,
        "seed": config.SEED,
        "early_stopping": config.EARLY_STOPPING,
        "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
    }
    unknown = sorted(set(overrides) - set(base))
    if unknown:
        raise ValueError(f"unknown hyperparameter(s): {unknown}")
    return {**base, **clean_overrides(overrides)}


def hyperparameter_hash(settings):
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def build_text_dataset(rows, eos_token):
    if not rows:
        raise ValueError("no rows to build a dataset from")
    texts = []
    for row in rows:
        try:
            texts.append(
                data.format_example(row["question"], row.get("chain") or "", eos_token)
            )
        except ValueError as e:
            raise ValueError(f"row {row['id']!r}: {e}") from e
    from datasets import Dataset

    return Dataset.from_dict({"text": texts})


def load_model_for_training(resume_adapter=None):
    from unsloth import FastLanguageModel

    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=resume_adapter or config.MODEL_NAME,
        max_seq_length=config.MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    peft_model = FastLanguageModel.get_peft_model(
        base_model,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        target_modules=config.LORA_TARGET_MODULES,
        use_gradient_checkpointing="unsloth",
        random_state=config.SEED,
    )
    return peft_model, tokenizer


def save_adapter(trainer, tokenizer, output_dir):
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def _model_name(name):
    return name.lower().removesuffix("-bnb-4bit")


def read_adapter_config(path):
    config_path = Path(path) / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} not found")
    return json.loads(config_path.read_text(encoding="utf-8"))


def check_adapter(path):
    adapter = read_adapter_config(path)
    base = adapter.get("base_model_name_or_path") or ""
    if _model_name(base) != _model_name(config.MODEL_NAME):
        raise ValueError(
            f"adapter {path} was trained on base model {base!r}, "
            f"not {config.MODEL_NAME!r}"
        )
    if adapter.get("r") != config.LORA_R:
        raise ValueError(
            f"adapter {path} has rank {adapter.get('r')}, expected {config.LORA_R}"
        )


def load_adapter(rounds, root=None):
    if not rounds:
        raise ValueError("no rounds recorded; train round 0 first")
    numbers = sorted(record["round"] for record in rounds)
    if numbers != list(range(len(numbers))):
        raise ValueError(f"gap in the recorded round numbers: {numbers}")
    path = config.adapter_path(numbers[-1], root=root)
    check_adapter(path)
    return path


def build_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir,
    epochs=None,
    per_device_train_batch_size=None,
    gradient_accumulation_steps=None,
    early_stopping=None,
    early_stopping_patience=None,
    seed=None,
    extra_callbacks=(),
):
    if len(train_dataset) == 0:
        raise ValueError("train dataset is empty")
    if len(eval_dataset) == 0:
        raise ValueError("eval dataset is empty")
    settings = hyperparameters(
        epochs=epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        seed=seed,
    )
    from transformers import EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth.chat_templates import train_on_responses_only

    callbacks = []
    if settings["early_stopping"]:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=settings["early_stopping_patience"]
            )
        )
    callbacks.extend(extra_callbacks)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            output_dir=output_dir,
            num_train_epochs=settings["epochs"],
            per_device_train_batch_size=settings["per_device_train_batch_size"],
            per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
            gradient_accumulation_steps=settings["gradient_accumulation_steps"],
            learning_rate=settings["learning_rate"],
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataset_text_field="text",
            max_length=config.MAX_SEQ_LENGTH,
            seed=settings["seed"],
            report_to="none",
        ),
        callbacks=callbacks,
    )
    return train_on_responses_only(
        trainer,
        instruction_part=config.INSTRUCTION_MARKER,
        response_part=config.RESPONSE_MARKER,
    )


class EpochEvaluator:
    def __init__(self, eval_rows, generate_fn, make_prompt=None):
        if not eval_rows:
            raise ValueError("eval rows are required")
        self.eval_rows = eval_rows
        self.generate_fn = generate_fn
        self.make_prompt = make_prompt or evaluation.finetuned_prompt
        self.epochs = []

    def evaluate_epoch(self, model, epoch, train_loss, eval_loss):
        from unsloth import FastLanguageModel

        FastLanguageModel.for_inference(model)
        try:
            _, metrics = evaluation.evaluate(
                self.eval_rows, self.generate_fn, self.make_prompt
            )
        finally:
            FastLanguageModel.for_training(model)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "accuracy": metrics["accuracy"],
            "format_compliance": metrics["format_compliance"],
            "avg_response_chars": metrics["avg_response_chars"],
            "per_category": {
                category: cell["accuracy"]
                for category, cell in metrics["per_category"].items()
            },
        }
        self.epochs.append(record)
        return record


def make_epoch_eval_callback(evaluator):
    from transformers import TrainerCallback

    class EpochEvalCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
            train_loss = next(
                (e["loss"] for e in reversed(state.log_history) if "loss" in e), None
            )
            eval_loss = (metrics or {}).get("eval_loss")
            evaluator.evaluate_epoch(model, round(state.epoch), train_loss, eval_loss)

    return EpochEvalCallback()


def build_training_record(
    *,
    round_n,
    trainer_state,
    train_rows,
    eval_rows,
    requested_epochs,
    epoch_results,
    versions,
    train_seconds,
    gpu_summary,
    peak_torch_memory_allocated_mib,
    hyperparams,
):
    epochs_run = trainer_state.epoch
    return {
        "round": round_n,
        "model_name": config.MODEL_NAME,
        "gpu_type": config.GPU_TYPE,
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "requested_epochs": requested_epochs,
        "epochs_run": epochs_run,
        "steps_run": trainer_state.global_step,
        "stopped_early": epochs_run < requested_epochs,
        "best_model_checkpoint": trainer_state.best_model_checkpoint,
        "best_metric": trainer_state.best_metric,
        "train_seconds": train_seconds,
        "versions": versions,
        "gpu_summary": gpu_summary,
        "peak_torch_memory_allocated_mib": peak_torch_memory_allocated_mib,
        "log_history": list(trainer_state.log_history),
        "epoch_results": epoch_results,
        "hyperparameters": hyperparams,
        "hyperparameter_hash": hyperparameter_hash(hyperparams),
    }


def read_rounds(path):
    path = Path(path)
    if not path.exists():
        return []
    try:
        rounds = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise ValueError(f"{path} is not valid JSON") from e
    if not isinstance(rounds, list):
        raise ValueError(  # noqa: TRY004 - bad data, one exception type
            f"{path} must hold a list of rounds"
        )
    return rounds


def record_round(path, record):
    rounds = read_rounds(path)
    numbers = [item["round"] for item in rounds]
    if record["round"] in numbers:
        raise ValueError(f"round {record['round']} is already recorded in {path}")
    if record["round"] != len(numbers):
        raise ValueError(
            f"gap in the round numbers: the next round must be {len(numbers)}, "
            f"got {record['round']}"
        )
    evaluation.save_result(path, [*rounds, record])


def response_start_index(tokenizer, question):
    prompt = data.format_prompt(question)
    return len(data.encode_ids(tokenizer, prompt))


def expected_trained_text(tokenizer, row):
    example = data.format_example(row["question"], row["chain"], config.EOS_TOKEN)
    ids = data.encode_ids(tokenizer, example)
    start = response_start_index(tokenizer, row["question"])
    return tokenizer.decode(ids[start:], skip_special_tokens=False)


def check_masking(tokenizer, rows, examples):
    if len(rows) != len(examples):
        raise ValueError(f"{len(rows)} rows but {len(examples)} examples")
    for row, example in zip(rows, examples, strict=True):
        kept = [
            token
            for token, label in zip(
                example["input_ids"], example["labels"], strict=True
            )
            if label != -100
        ]
        got = tokenizer.decode(kept, skip_special_tokens=False)
        expected = expected_trained_text(tokenizer, row)
        if got != expected:
            raise ValueError(
                f"row {row['id']!r}: the unmasked text {got!r} is not the "
                f"response and end token {expected!r}"
            )


def _as_lists(values):
    return values.tolist() if hasattr(values, "tolist") else [list(r) for r in values]


def unmasked_runs(ids, labels):
    runs, current = [], []
    for token, label in zip(ids, labels, strict=True):
        if label != -100:
            current.append(token)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def check_batch_masking(tokenizer, rows, batch, expected_count=None):
    expected = {expected_trained_text(tokenizer, row) for row in rows}
    runs = []
    for ids, labels in zip(
        _as_lists(batch["input_ids"]), _as_lists(batch["labels"]), strict=True
    ):
        runs.extend(unmasked_runs(ids, labels))
    if not runs:
        raise ValueError("no trained tokens in the batch")
    if expected_count is not None and len(runs) != expected_count:
        raise ValueError(
            f"{len(runs)} trained runs but {expected_count} examples in the batch"
        )
    for run in runs:
        text = tokenizer.decode(run, skip_special_tokens=False)
        if text not in expected:
            raise ValueError(
                f"a trained run does not match any response and end token: {text!r}"
            )


def plan_round(round_n, rounds, settings, resume_adapter="", root="/checkpoints"):
    if round_n < 0:
        raise ValueError("round numbers start at 0")
    if round_n > config.MAX_ROUNDS:
        raise ValueError(
            f"round {round_n} is beyond the maximum of {config.MAX_ROUNDS} targeted rounds"
        )
    numbers = sorted(record["round"] for record in rounds)
    if round_n in numbers:
        raise ValueError(f"round {round_n} is already recorded")
    if numbers != list(range(round_n)):
        raise ValueError(
            f"gap in the round numbers: recorded {numbers}, so the next round "
            f"must be {len(numbers)}"
        )
    if round_n == 0:
        if resume_adapter:
            raise ValueError(
                "round 0 starts from the base model; do not pass a resume adapter"
            )
        return None
    expected = str(config.adapter_path(round_n - 1, root=root))
    if resume_adapter and resume_adapter != expected:
        raise ValueError(
            f"wrong resume adapter {resume_adapter!r}; round {round_n} continues "
            f"from {expected!r}"
        )
    if rounds[0].get("hyperparameter_hash") != hyperparameter_hash(settings):
        raise ValueError(
            "hyperparameters changed since round 0; every round must use the same ones"
        )
    return expected


def round_decision(weak_categories):
    return "continue" if weak_categories else "stop"


def best_epoch(epoch_results):
    if not epoch_results:
        raise ValueError("no epochs to choose from")
    scored = [e for e in epoch_results if e.get("eval_loss") is not None]
    if not scored:
        return epoch_results[-1]
    return min(scored, key=lambda e: (e["eval_loss"], e["epoch"]))
