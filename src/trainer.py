"""
Fine-tune Llama 3.2 1B on Star Wars dialogue using Unsloth (for fast model
loading/patching) with LoRA, feeding pre-tokenized input_ids directly
rather than decoding chunks back to text and re-tokenizing them -- that
round-trip isn't guaranteed to preserve exact token counts and caused a
length mismatch between input and label tensors.
"""

import time

import torch
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from src.dataloader import load_scenes, scenes_to_text, split_scenes_by_movie

MODEL_ID = "unsloth/Llama-3.2-1B"
MAX_SEQ_LENGTH = 128
OUTPUT_DIR = "outputs/star_wars_lora"


def load_training_text(data_path="data/star_wars_script.jsonl"):
    """Load and flatten the training split (everything except val/test titles)."""
    scenes = load_scenes(data_path)
    train_scenes, _, _ = split_scenes_by_movie(scenes)
    return scenes_to_text(train_scenes)


def load_model_for_training(
    model_id=MODEL_ID, max_seq_length=MAX_SEQ_LENGTH
):  # pragma: no cover
    """Load the base model and tokenizer, wrapped with LoRA adapters via Unsloth."""
    from unsloth import FastLanguageModel  # imported here, not at module level,

    # so this module can be imported in CI/tests without requiring Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )

    model = FastLanguageModel.get_peft_model(
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
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=17,
    )
    return model, tokenizer


def build_training_dataset(tokenizer, text, max_seq_length=MAX_SEQ_LENGTH):
    """
    Chunk the flattened training text into fixed-length token sequences,
    keeping the token IDs directly rather than decoding back to text --
    this avoids a lossy decode/re-encode round-trip that can silently
    change the token count of a chunk.
    """
    from datasets import Dataset

    token_ids = tokenizer(text, return_tensors="pt").input_ids[0]

    examples = []
    stride = max_seq_length // 4
    for i in range(0, len(token_ids) - max_seq_length, stride):
        chunk = token_ids[i : i + max_seq_length]
        examples.append({"input_ids": chunk.tolist()})

    return Dataset.from_list(examples)


def train_model(
    model_id=MODEL_ID,
    data_path="data/star_wars_script.jsonl",
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=4,
):  # pragma: no cover
    """Full training pipeline: load model, load training data, fine-tune with LoRA."""
    model, tokenizer = load_model_for_training(model_id)

    train_text = load_training_text(data_path)
    train_dataset = build_training_dataset(tokenizer, train_text)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=num_train_epochs,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start_time
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3

    return {
        "model": model,
        "tokenizer": tokenizer,
        "train_loss": train_result.training_loss,
        "training_time_seconds": elapsed,
        "peak_vram_gb": peak_vram_gb,
    }


if __name__ == "__main__":  # pragma: no cover
    results = train_model()
    print(f"Final training loss: {results['train_loss']:.4f}")
    print(f"Training time: {results['training_time_seconds']:.1f} seconds")
    print(f"Peak VRAM used: {results['peak_vram_gb']:.2f} GB")

    results["model"].save_pretrained(OUTPUT_DIR)
    results["tokenizer"].save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")
