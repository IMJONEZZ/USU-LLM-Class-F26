"""
Fine-tune Llama 3.2 1B Instruct on a custom dataset of trick-question-style
reasoning examples, using Unsloth and LoRA. Training data combines 215
misconception-style examples from TruthfulQA with 24 synthetic examples
covering categories (scaling traps, letter counting, trick math, wordplay,
spatial reasoning, scene inference, mythology, riddles) generated locally
with a Llama 3.2 3B Instruct model via distilabel, after Hugging Face's
Inference API proved infeasible (a version-incompatibility bug in
distilabel's InferenceEndpointsLLM, and separately, the free tier's $0.10/
month credit limit being exhausted with no pay-as-you-go option for free
accounts).
"""

import json
import time

import torch
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"
DATA_PATH = "data/trick_question_training.jsonl"
MAX_SEQ_LENGTH = 256
OUTPUT_DIR = "outputs/trick_question_lora"


def load_training_examples(data_path=DATA_PATH):
    """Load the combined (TruthfulQA + synthetic) training examples from the jsonl file."""
    examples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def load_model_for_training(
    model_id=MODEL_ID, max_seq_length=MAX_SEQ_LENGTH
):  # pragma: no cover
    """Load the base Instruct model and tokenizer, wrapped with LoRA adapters via Unsloth."""
    from unsloth import FastLanguageModel  # imported here so this module can be
    # imported in CI/tests without requiring Unsloth or a GPU

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


def build_training_dataset(tokenizer, examples, max_seq_length=MAX_SEQ_LENGTH):
    """
    Format each (question, answer) example using the tokenizer's chat template,
    so the model is trained in the same conversational format it will be
    asked questions in later (via answers.ipynb).
    """
    from datasets import Dataset

    formatted_examples = []
    for ex in examples:
        messages = [
            {"role": "user", "content": ex["question"]},
            {"role": "assistant", "content": ex["answer"]},
        ]
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            max_length=max_seq_length,
            truncation=True,
        )
        formatted_examples.append({"input_ids": token_ids})

    return Dataset.from_list(formatted_examples)


def train_model(
    model_id=MODEL_ID,
    data_path=DATA_PATH,
    output_dir=OUTPUT_DIR,
    num_train_epochs=8,
    per_device_train_batch_size=4,
):  # pragma: no cover
    """Full training pipeline: load model, load training data, fine-tune with LoRA."""
    model, tokenizer = load_model_for_training(model_id)

    examples = load_training_examples(data_path)
    train_dataset = build_training_dataset(tokenizer, examples)

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
