"""Fine-tune a pretrained causal language model with Unsloth LoRA."""

from src.dataloader import create_datasets
from src.evaluator import run_evaluation

MODEL_NAME = "meta-llama/Llama-3.2-1B"


def run_training(
    model_name: str = MODEL_NAME,
    output_dir: str = "models/star-wars-llama",
    block_size: int = 256,
    num_train_epochs: int = 3,
) -> dict[str, float | None]:
    """Train a LoRA adapter and return runtime, VRAM, and validation-loss metrics."""
    import torch
    import unsloth
    from transformers import (
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=block_size,
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    datasets = create_datasets(tokenizer, block_size=block_size)
    model = unsloth.FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        fp16=(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
        bf16=(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        processing_class=tokenizer,
    )
    initial_eval_loss = trainer.evaluate()["eval_loss"]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    train_result = trainer.train()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_vram_allocated_gib = torch.cuda.max_memory_allocated() / 1024**3
        peak_vram_reserved_gib = torch.cuda.max_memory_reserved() / 1024**3
    else:
        peak_vram_allocated_gib = None
        peak_vram_reserved_gib = None

    final_eval_loss = trainer.evaluate()["eval_loss"]
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    metrics = {
        "training_runtime_seconds": train_result.metrics["train_runtime"],
        "initial_validation_loss": initial_eval_loss,
        "final_validation_loss": final_eval_loss,
        "validation_loss_improvement": initial_eval_loss - final_eval_loss,
        "validation_loss_improvement_percent": (
            (initial_eval_loss - final_eval_loss) / initial_eval_loss * 100
        ),
        "peak_vram_allocated_gib": peak_vram_allocated_gib,
        "peak_vram_reserved_gib": peak_vram_reserved_gib,
    }
    print(f"Training metrics: {metrics}")
    return metrics


if __name__ == "__main__":  # pragma: no cover
    print("Base model:")
    print(run_evaluation(model_name=MODEL_NAME))

    run_training()

    print("Fine-tuned model:")
    print(run_evaluation(model_name=MODEL_NAME, adapter_path="models/star-wars-llama"))
