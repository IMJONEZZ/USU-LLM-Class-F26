from datasets import Dataset
from transformers import EarlyStoppingCallback

from src.config import (
    EARLY_STOPPING_PATIENCE,
    EVAL_STEPS,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_R,
    LORA_TARGET_MODULES,
    MAX_SEQ_LENGTH,
    MAX_STEPS,
    MODEL_NAME,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    SEED,
)
from src.prompts import INSTRUCTION_MARKER, RESPONSE_MARKER, format_example


def load_model_for_training():
    from unsloth import FastLanguageModel

    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    peft_model = FastLanguageModel.get_peft_model(
        base_model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    return peft_model, tokenizer


def build_text_dataset(resumes, categories, eos_token):
    texts = [
        format_example(resume_text, category, categories) + eos_token
        for resume_text, category in zip(resumes["Resume_str"], resumes["Category"])
    ]
    return Dataset.from_dict({"text": texts})


def make_early_stopping_callback():
    return EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)


def build_trainer(model, tokenizer, train_dataset, eval_dataset, output_dir):
    from trl import SFTConfig, SFTTrainer
    from unsloth.chat_templates import train_on_responses_only

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            max_steps=MAX_STEPS,
            eval_strategy="steps",
            eval_steps=EVAL_STEPS,
            save_strategy="steps",
            save_steps=EVAL_STEPS,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            dataset_text_field="text",
            max_length=MAX_SEQ_LENGTH,
        ),
        callbacks=[make_early_stopping_callback()],
    )
    return train_on_responses_only(
        trainer,
        instruction_part=INSTRUCTION_MARKER,
        response_part=RESPONSE_MARKER,
    )
