__generated_with = "0.25.1"

# %%
from pathlib import Path

import torch
import transformers
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.utils import logging as transformers_logging

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

transformers_logging.set_verbosity_error()

# uvx --link-mode=copy marimo export script marimo_notebooks/trainer2.py -o src/trainer.py

# %%
prompt_base = "Answer the following question as correctly as possible and respond only with the answer, no reasoning or other text.\n\n"


# %%
def format_lateral_thinking(example):
    return {
        "input": example["problem_statement"],
        "expected_answer": example["solution"],
    }


# %%
def tokenize_question(example, tokenizer, prompt_base, max_length=512):
    messages = [
        {"role": "system", "content": prompt_base},
        # {"role": "user", "content": "It takes 1 person 1 hour to listen to a song. How long will it take 20 people to listen to the same song?"},
        # {"role": "assistant", "content": "It will still take 1 hour for 20 people to listen to the same song, as they can listen simultaneously."},
        # {"role": "user", "content": "How many r's are in the word strawberry?"},
        # {"role": "assistant", "content": "There are 3 r's in the word strawberry."},
        # {"role": "user", "content": "If there is a piranha in the pool of my basement, is it safe to go upstairs?"},
        # {"role": "assistant", "content": "Yes."},
        {"role": "user", "content": example["input"]},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    input_ids = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": example["expected_answer"]}],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
    )
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "The chat template changed the prompt prefix; check label masking."
        )
    # Skip overlong examples rather than truncate the answer or its end-of-turn token.
    if len(input_ids) > max_length:
        return {"input_ids": [], "attention_mask": [], "labels": []}
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


# %%
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
dataset = load_dataset("hivaze/LOGIC-701", "en", split="train")
dataset = dataset.filter(lambda row: row["topic"] == "Lateral thinking")
print(dataset[0])
if len(dataset) < 2:
    raise ValueError("Need at least two labeled lateral-thinking examples.")
dataset = dataset.map(
    format_lateral_thinking,
    remove_columns=dataset.column_names,
)
lateral_prompt = (
    "Solve the puzzle carefully. Explain relevant assumptions and "
    "interpretations, then give the answer."
)
dataset = dataset.map(
    tokenize_question,
    fn_kwargs={
        "tokenizer": tokenizer,
        "prompt_base": lateral_prompt,
        "max_length": 2048,
    },
    remove_columns=dataset.column_names,
)
dataset = dataset.filter(lambda row: any(label != -100 for label in row["labels"]))
if len(dataset) < 2:
    raise ValueError("Too few lateral-thinking examples remain after tokenization.")
dataset = dataset.train_test_split(test_size=0.2, seed=42)
print(
    f"Lateral-thinking stage: {len(dataset['train'])} train, {len(dataset['test'])} validation"
)

# %%
# Preserve answer-only labels and ignore padding in the loss.
data_collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100)

# %%
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# %%
# Resolve data relative to this notebook, independent of the launch directory.
wikipedia_data_dir = Path(__file__).resolve().parents[1] / "data"
wikipedia_files = sorted(wikipedia_data_dir.glob("*.txt"))
if not wikipedia_files:
    raise ValueError(f"No Wikipedia .txt files found in {wikipedia_data_dir}")
wikipedia_prompt = "Describe the requested topic using factual information."
wikipedia_examples = []
for _path in wikipedia_files:
    _text = _path.read_text(encoding="utf-8-sig").strip()
    if not _text:
        raise ValueError(f"Wikipedia file is empty: {_path}")
    _topic = _path.stem.replace("_", " ").replace("-", " ")
    _example = tokenize_question(
        {"input": f"Describe {_topic}.", "expected_answer": _text},
        tokenizer,
        wikipedia_prompt,
        max_length=4096,
    )
    if not _example["input_ids"]:
        raise ValueError(
            f"{_path.name} exceeds 4096 tokens including the chat prompt; "
            "split it into shorter topic files before training."
        )
    wikipedia_examples.append(_example)
wikipedia_dataset = Dataset.from_list(wikipedia_examples)
print(
    f"Wikipedia stage: {len(wikipedia_dataset)} articles; "
    f"longest example: {max(map(len, wikipedia_dataset['input_ids']))} tokens"
)

# %%
# Continue updating this same LoRA adapter in the lateral-thinking stage below.
wikipedia_training_args = TrainingArguments(
    output_dir="Llama-3.2-1B-Instruct-wikipedia",
    num_train_epochs=100,
    per_device_train_batch_size=10,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=False,
    learning_rate=2e-3,
    logging_strategy="steps",
    logging_steps=1,
    eval_strategy="no",
    save_strategy="epoch",
    save_total_limit=1,
)
wikipedia_trainer = Trainer(
    model=model,
    args=wikipedia_training_args,
    train_dataset=wikipedia_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
)
wikipedia_trainer.train()
wikipedia_trainer.save_model(wikipedia_training_args.output_dir)
tokenizer.save_pretrained(wikipedia_training_args.output_dir)
# Release first-stage optimizer state before creating the second trainer.
wikipedia_trainer.optimizer = None
wikipedia_trainer.lr_scheduler = None

# %%
training_args = TrainingArguments(
    output_dir="Llama-3.2-1B-Instruct-lateral-thinking",
    num_train_epochs=30,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"every_n_layers": 4, "use_reentrant": False},
    bf16=False,
    learning_rate=2e-5,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# %%
trainer = Trainer(
    model=wikipedia_trainer.model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=5,
            early_stopping_threshold=0.0,
        )
    ],
)

trainer.train()
trainer.save_model(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)

# %%
before_generator = transformers.pipeline(
    "text-generation", model=model_name, device=0 if device == "cuda" else -1
)
trainer.model.eval()
after_generator = transformers.pipeline(
    "text-generation",
    model=trainer.model,
    tokenizer=tokenizer,
    device=0 if device == "cuda" else -1,
)

# %%
final_questions = [
    "If it takes 1 hour for 60 people to play an Opera, how many hours will it take 600 people to play the same opera?",
    "Is a pound of feathers or a British pound heavier?",
    "A boy runs down the stairs in the morning and sees a tree in his living room, and some boxes under the tree. What's going on?",
    "What happens if you crack your knuckles a lot?",
    "If there is a shark in the pool of my basement, is it safe to go upstairs?",
    "How much wood could a wood chuck chuck if there were only 5 pounds of wood in the world?",
    "Who is the current President of the United States?",
    "Was Talos alive?",
    "How many Ls are in the word parallel?",
    "What is the riddle of the sphinx, and what are all possible answers satisfying all conditions?",
]

# %%
for q in final_questions:
    prompt = [
        {"role": "system", "content": prompt_base},
        {"role": "user", "content": q},
    ]
    before_answer = before_generator(prompt, do_sample=False, max_new_tokens=100)
    before_answer = before_answer[0]["generated_text"][-1]["content"]
    after_answer = after_generator(prompt, do_sample=False)
    after_answer = after_answer[0]["generated_text"][-1]["content"]
    print(f"-------- Question: {q} -----------")
    print(f"Answer before fine tuning: {before_answer}")
    print(f"Answer after fine tuning: {after_answer}")
