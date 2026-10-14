"""Fine-tune a Llama 3.2 LoRA adapter on Modal using the combined dataset."""

import json
import os
from pathlib import Path
from typing import Any

from datasets import load_dataset

import modal

DATASET_ID = "domenicrosati/TruthfulQA"
TRAINING_ROW_LIMIT = 200
MANUAL_DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "manual_qa.template.jsonl"
)
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MAX_SEQ_LENGTH = 1024
OUTPUT_VOLUME_NAME = "llama32-1b-lora-output"
OUTPUT_MOUNT_PATH = "/outputs"

# Install Unsloth and its compatible training stack in the remote GPU image.
# Import Unsloth before Transformers/TRL in the function below so its patches
# are applied before those libraries initialize.
training_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "accelerate==1.9.0",
    "datasets==3.6.0",
    "huggingface_hub==0.34.2",
    "peft==0.16.0",
    "transformers==4.54.0",
    "trl==0.19.1",
    "unsloth[cu128-torch270]==2025.7.8",
    "unsloth_zoo==2025.7.10",
)

app = modal.App("llama32-1b-lora-training")
output_volume = modal.Volume.from_name(
    OUTPUT_VOLUME_NAME,
    create_if_missing=True,
)


def load_manual_examples(
    dataset_path: str | Path = MANUAL_DATASET_PATH,
) -> list[dict[str, Any]]:
    """Load custom question-answer examples from a JSONL file."""
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manual examples file not found: {path}")

    examples = []
    with path.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} on line {line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise TypeError(
                    f"Manual example on line {line_number} must be a JSON object."
                )

            question = record.get("question")
            answer = record.get("answer")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(
                    f"Missing question in manual example on line {line_number}."
                )
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(
                    f"Missing answer in manual example on line {line_number}."
                )

            examples.append(
                {
                    "task_type": "manual",
                    "messages": [
                        {"role": "user", "content": question.strip()},
                        {"role": "assistant", "content": answer.strip()},
                    ],
                }
            )
    return examples


def load_training_examples(
    dataset_id: str | None = DATASET_ID,
    row_limit: int | None = None,
    manual_dataset_path: str | Path | None = MANUAL_DATASET_PATH,
) -> list[dict[str, Any]]:
    """Combine enabled sources into chat-formatted question-answer examples."""
    if row_limit is None:
        row_limit = TRAINING_ROW_LIMIT
    if row_limit < 1:
        raise ValueError("row_limit must be at least 1")
    if dataset_id is None and manual_dataset_path is None:
        raise ValueError("At least one training data source must be enabled.")

    examples = []
    if dataset_id is not None:
        dataset = load_dataset(dataset_id, split=f"train[:{row_limit}]")
        required_columns = {"Question", "Best Answer"}
        missing_columns = required_columns.difference(dataset.column_names)
        if missing_columns:
            raise ValueError(
                f"Dataset {dataset_id} is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        for row_number, record in enumerate(dataset, start=1):
            question = record["Question"]
            answer = record["Best Answer"]
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Missing question in dataset row {row_number}.")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"Missing Best Answer in dataset row {row_number}.")

            examples.append(
                {
                    "task_type": "truthfulqa",
                    "messages": [
                        {"role": "user", "content": question.strip()},
                        {"role": "assistant", "content": answer.strip()},
                    ],
                }
            )

        if len(examples) != row_limit:
            raise ValueError(
                f"Expected {row_limit} rows from {dataset_id}, got {len(examples)}."
            )

    if manual_dataset_path is not None:
        examples.extend(load_manual_examples(manual_dataset_path))
    return examples


@app.function(
    image=training_image,
    gpu="T4",
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={OUTPUT_MOUNT_PATH: output_volume},
    env={"HF_HOME": "/root/.cache/huggingface"},
)
def train_lora(training_examples: list[dict[str, Any]], run_name: str) -> str:
    """Train and save a 4-bit LoRA adapter on a Modal T4 GPU."""
    import torch
    import unsloth  # noqa: F401  (must run before transformers/trl imports)
    from datasets import Dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "The Modal 'huggingface' secret must contain the HF_TOKEN variable."
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
        token=os.environ["HF_TOKEN"],
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
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    formatted_rows = [
        {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }
        for example in training_examples
    ]
    dataset = Dataset.from_list(formatted_rows)
    run_directory = Path(OUTPUT_MOUNT_PATH) / run_name
    run_directory.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        args=TrainingArguments(
            output_dir=str(run_directory / "checkpoints"),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=5,
            learning_rate=2e-4,
            warmup_ratio=0.05,
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            report_to="none",
            save_strategy="no",
        ),
    )
    train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )
    trainer.train()

    adapter_directory = run_directory / "adapter"
    model.save_pretrained(str(adapter_directory))
    tokenizer.save_pretrained(str(adapter_directory))
    manifest = {
        "model": MODEL_ID,
        "training_examples": len(training_examples),
        "epochs": 3,
        "max_seq_length": MAX_SEQ_LENGTH,
        "adapter_path": str(adapter_directory),
    }
    (run_directory / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    output_volume.commit()
    return f"Saved LoRA adapter to {adapter_directory}"


@app.local_entrypoint()
def main(run_name: str = "truthfulqa-llama32-1b-lora") -> None:
    """Load the combined data locally and start the remote GPU training job."""
    training_examples = load_training_examples()
    print(
        f"Starting training with {len(training_examples)} examples "
        f"({TRAINING_ROW_LIMIT} TruthfulQA rows plus manual examples)."
    )
    print(train_lora.remote(training_examples, run_name))


if __name__ == "__main__":
    training_examples = load_training_examples()
    manual_count = len(load_manual_examples())
    print(
        f"Loaded {len(training_examples)} training examples total: "
        f"{TRAINING_ROW_LIMIT} from {DATASET_ID} using Best Answer and "
        f"{manual_count} manual examples."
    )
