import os
from pathlib import Path

try:
    import modal
except ModuleNotFoundError:  # Modal is only needed when launching remote training.
    modal = None


def tokenize_example(example, tokenizer, max_length: int = 512) -> dict:
    """Tokenize an instruction pair and mask prompt tokens from the loss."""
    prompt = f"Convert to PPA:\n{example['input']}\nOutput:\n"
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    output_ids = tokenizer(
        example["output"],
        add_special_tokens=False,
    )["input_ids"]
    output_ids.append(tokenizer.eos_token_id)

    if len(output_ids) > max_length:
        output_ids = output_ids[: max_length - 1] + [tokenizer.eos_token_id]
    prompt_budget = max_length - len(output_ids)
    prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []
    input_ids = prompt_ids + output_ids

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + output_ids,
    }


def _train_model():
    """Run training; dependencies are imported here for offline testability."""
    import os

    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    model_id = "meta-llama/Llama-3.2-1B"
    max_length = 512

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = load_dataset(
        "csv",
        data_files="/training_data/generate_data.csv",
        split="train",
    )
    splits = dataset.train_test_split(test_size=0.2, seed=42)
    tokenized_splits = splits.map(
        lambda example: tokenize_example(example, tokenizer, max_length),
        remove_columns=dataset.column_names,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    training_args = TrainingArguments(
        output_dir="/outputs/lora_training",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_splits["train"],
        eval_dataset=tokenized_splits["test"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    eval_results = trainer.evaluate()

    adapter_path = "/outputs/lora_adapter"
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    if modal is not None:
        output_volume.commit()

    hf_token = os.environ["HF_TOKEN"]
    hf_api = HfApi(token=hf_token)
    username = hf_api.whoami()["name"]
    repo_id = f"{username}/ppa-llama-3.2-1b-lora"
    hf_api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=True,
        exist_ok=True,
        token=hf_token,
    )
    hf_api.upload_folder(
        repo_id=repo_id,
        folder_path=adapter_path,
        token=hf_token,
        commit_message="Upload PPA LoRA adapter",
    )

    return {
        "repo_id": repo_id,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_results,
    }


if modal is not None and "HF_TOKEN" in os.environ:
    app = modal.App("gpu-test")
    csv_path = (
        Path(__file__).resolve().parent.parent / "training_data" / "generate_data.csv"
    )
    output_volume = modal.Volume.from_name("ppa-lora-output", create_if_missing=True)
    hf_secret = modal.Secret.from_local_environ(["HF_TOKEN"])
    image = (
        modal.Image.debian_slim()
        .pip_install(
            "torch",
            "transformers",
            "datasets",
            "peft",
            "accelerate",
            "huggingface_hub",
        )
        .add_local_file(str(csv_path), remote_path="/training_data/generate_data.csv")
    )
    train_model = app.function(
        image=image,
        gpu="T4",
        volumes={"/outputs": output_volume},
        secrets=[hf_secret],
    )(_train_model)
