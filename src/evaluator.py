__generated_with = "0.24.2"

# %%
import numpy as np
import pandas as pd
import torch
import transformers
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers.utils import logging as transformers_logging

transformers_logging.set_verbosity_error()
# uvx --link-mode=copy marimo export script marimo_notebooks/evaluator.py -o src/evaluator.py

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# %%
riddles = pd.read_csv(
    "C:/Users/conno/OneDrive/Documents/USU PHD/USU-LLM-Class-F26/data/riddles.csv"
)
riddles = riddles[["Riddle", "Answer"]].rename(
    columns={"Riddle": "input", "Answer": "expected_answer"}
)
# %%
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

# %%
# generator = transformers.pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
# generator = transformers.pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")
generator = transformers.pipeline(
    "text-generation", model="HuggingFaceTB/SmolLM2-360M-Instruct"
)

prompt = [
    {
        "role": "user",
        "content": "Answer this riddle with only the answer: "
        "What has keys but cannot open locks?",
    }
]

result = generator(
    prompt,
    do_sample=True,
    temperature=1,
    top_p=0.9,
)

print(result[0]["generated_text"][1]["content"])


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
            {"role": "user", "content": "What has hands but cannot clap?"},
            {"role": "assistant", "content": "Clock"},
            {
                "role": "user",
                "content": "What has a thumb and four fingers but is not alive?",
            },
            {"role": "assistant", "content": "Glove"},
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
prompt_base = (
    "Solve the user's riddle. Respond with only the name of the answer "
    "in 1 to 3 words. Omit introductory text, explanations, and quotation marks."
)
# HuggingfaceTB/SmolLM2-360M-Instruct
hf360m = evaluate(
    transformers.pipeline(
        "text-generation",
        model="HuggingFaceTB/SmolLM2-360M-Instruct",
        device=0 if device == "cuda" else -1,
    ),
    eval_df=riddles,
    embedder=embedder,
    prompt_base=prompt_base,
)
# Qwen/Qwen2.5-0.5B-Instruct
qwen_05b = evaluate(
    transformers.pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        device=0 if device == "cuda" else -1,
    ),
    eval_df=riddles,
    embedder=embedder,
    prompt_base=prompt_base,
)
# Qwen/Qwen2.5-1.5B-Instruct
qwen_15b = evaluate(
    transformers.pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        device=0 if device == "cuda" else -1,
    ),
    eval_df=riddles,
    embedder=embedder,
    prompt_base=prompt_base,
)
print("HuggingFaceTB/SmolLM2-360M-Instruct:", hf360m[0])


print("Qwen/Qwen2.5-0.5B-Instruct:", qwen_05b[0])

print("Qwen/Qwen2.5-1.5B-Instruct:", qwen_15b[0])
