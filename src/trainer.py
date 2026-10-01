__generated_with = "0.25.0"

# %%
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from transformers.utils import logging as transformers_logging

transformers_logging.set_verbosity_error()
# uvx --link-mode=copy marimo export script marimo_notebooks/trainer.py -o src/trainer.py

# %%
prompt_base = (
    "Solve the user's riddle. Respond with only the name of the answer "
    "in 1 to 3 words. Omit introductory text, explanations, and quotation marks."
)


# %%
def tokenize_riddle(example, tokenizer, prompt_base, max_length=512):
    messages = [
        {"role": "system", "content": prompt_base},
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
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
dataset = load_dataset("csthomps/riddles", split="train")

dataset = dataset.map(
    tokenize_riddle,
    fn_kwargs={"tokenizer": tokenizer, "prompt_base": prompt_base},
    remove_columns=dataset.column_names,
)
dataset = dataset.filter(lambda row: any(label != -100 for label in row["labels"]))
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# %%
# Preserve answer-only labels and ignore padding in the loss.
data_collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100)

# %%
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")

# %%
training_args = TrainingArguments(
    output_dir="Qwen2.5-1.5B-Instruct-riddles",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"every_n_layers": 4},
    bf16=False,
    learning_rate=2e-5,
    logging_strategy="steps",
    logging_steps=1,
    eval_strategy="steps",
    eval_steps=1,
    save_strategy="steps",
    save_steps=1,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# %%
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)


# %%
def evaluate(generator, eval_df, embedder, prompt_base):
    # generator: The huggingface model to evaluate
    # eval_df: A pandas dataframe containing evaluation data.  Specifically it should have 'input' and 'expected_answer' columns
    # embedder: The sentence transformer model to use for embedding the riddles and answers

    similarities = []
    generated_answers = []

    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
        input = row["input"]
        expected_answer = row["expected_answer"]

        prompt = [
            {"role": "system", "content": prompt_base},
            {"role": "user", "content": input},
        ]

        result = generator(prompt, do_sample=False, max_new_tokens=16)
        generated_answer = result[0]["generated_text"][-1]["content"]

        # Compute embeddings for the expected answer and the generated answer
        embeddings = embedder.encode([expected_answer, generated_answer])
        cos_sim = np.dot(np.array(embeddings[0]), np.array(embeddings[1])) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        similarities.append(cos_sim)
        generated_answers.append(generated_answer)

    output_df = pd.DataFrame(
        {
            "input": eval_df["input"],
            "expected_answer": eval_df["expected_answer"],
            "generated_answer": generated_answers,
            "similarity": similarities,
        }
    )

    return np.mean(similarities), similarities, output_df


# %%
before_generator = transformers.pipeline(
    "text-generation", model=model_name, device=0 if device == "cuda" else -1
)
after_generator = transformers.pipeline(
    "text-generation",
    model=trainer.model,
    tokenizer=tokenizer,
    device=0 if device == "cuda" else -1,
)
test_data = pd.DataFrame(load_dataset("csthomps/riddles", split="test"))

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

before_results = evaluate(before_generator, test_data, embedder, prompt_base)
after_results = evaluate(after_generator, test_data, embedder, prompt_base)

print("Mean similarity before fine-tuning:", before_results[0])
print("Mean similarity after fine-tuning:", after_results[0])
