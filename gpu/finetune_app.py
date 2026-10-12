"""Runs the assignment 6 fine tune on a rented T4 through Modal.

Asks the stock model the ten graded questions, trains, then asks again, so the
before and after sit next to each other. The training logic is in
``src/trainer.py`` and the data is built by ``src/dataset.py``; this file is
the GPU wiring and nothing else.

It also asks a handful of trick questions from the MisguidedAttention set,
which nothing here was trained on. Those are the ones that tell me whether the
model learned the idea or just my phrasings.

    modal run gpu/finetune_app.py
"""

import json

import modal

GPU_TYPE = "T4"
CACHE_DIR = "/cache"

# Trick questions from cpldcpu/MisguidedAttention, which publishes them
# scrambled so they stay out of training sets. Held out, never trained on.
HELD_OUT = [
    "A man with his sheep wants to cross a river. He has a boat that can carry both him and the animal. How do both get to the other side of the river?",
    "You have two ropes, each of which takes exactly 60 minutes to burn completely. However, the ropes burn unevenly. You have no other timing device. How can you measure exactly 60 minutes using the ropes and matches to light them?",
    "A dead cat is placed into a box along with a nuclear isotope, a vial of poison and a radiation detector. If the radiation detector detects radiation, it will release the poison. The box is opened one day later. What is the probability of the cat being alive?",
    "In a room of 30 people, what's the probability that at least two do not share a birthday?",
    "You're on a game show and are presented with three doors. Behind one is a donkey, and behind the other two are luxury cars. You pick one, but before you can open it the host opens one of the others revealing a luxury car. He then offers you the choice of keeping your existing door or swapping to the other unrevealed one. What should you do to win a car?",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "unsloth==2025.3.19",
        "unsloth_zoo==2025.3.17",
        "transformers==4.49.0",
        "trl==0.15.2",
        "peft==0.14.0",
        "accelerate==1.4.0",
        "bitsandbytes==0.45.3",
        "datasets==3.3.2",
        "huggingface_hub[hf_transfer]==0.29.1",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": CACHE_DIR})
    .add_local_python_source("src")
    .add_local_file("data/paraphrases.json", "/root/paraphrases.json")
)

app = modal.App("hw6-finetune", image=image)
hf_cache = modal.Volume.from_name("hw6-hf-cache", create_if_missing=True)


@app.function(
    gpu=GPU_TYPE,
    timeout=5400,
    volumes={CACHE_DIR: hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run(epochs: int = 3, chat_rows: int = 300):
    import torch

    from src.dataset import (
        HOMEWORK_QUESTIONS,
        build_dataset,
        load_chat_rows,
        to_messages,
    )
    from src.trainer import answer, attach_lora, finetune, load_base_model

    print(f"gpu: {torch.cuda.get_device_name(0)}")

    with open("/root/paraphrases.json", encoding="utf-8") as handle:
        paraphrases = json.load(handle)

    chat = load_chat_rows(limit=chat_rows)
    rows = build_dataset(chat_rows=chat, paraphrases=paraphrases)
    counts = {}
    for row in rows:
        counts[row["source"]] = counts.get(row["source"], 0) + 1
    print(f"training rows: {len(rows)}")
    for source, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"   {source}: {n}")

    model, tokenizer = load_base_model()

    print("\n=== BEFORE ===")
    before = answer(model, tokenizer, HOMEWORK_QUESTIONS)
    for question, reply in zip(HOMEWORK_QUESTIONS, before):
        print(f"\nQ: {question}\nA: {reply}")

    before_held_out = answer(model, tokenizer, HELD_OUT)

    torch.cuda.reset_peak_memory_stats()
    model = attach_lora(model)
    stats = finetune(model, tokenizer, to_messages(rows), epochs=epochs)
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    print("\n=== AFTER ===")
    after = answer(model, tokenizer, HOMEWORK_QUESTIONS)
    for question, reply in zip(HOMEWORK_QUESTIONS, after):
        print(f"\nQ: {question}\nA: {reply}")

    after_held_out = answer(model, tokenizer, HELD_OUT)
    print("\n=== HELD OUT (never trained on) ===")
    for question, reply in zip(HELD_OUT, after_held_out):
        print(f"\nQ: {question[:90]}...\nA: {reply}")

    metrics = dict(stats.metrics)
    print(
        f"\ntrain_loss {metrics.get('train_loss')}, runtime {metrics.get('train_runtime')}s"
    )
    print(f"peak vram {peak_gb:.2f} GB")

    return {
        "questions": HOMEWORK_QUESTIONS,
        "before": before,
        "after": after,
        "held_out_questions": HELD_OUT,
        "held_out_before": before_held_out,
        "held_out_after": after_held_out,
        "rows": len(rows),
        "row_counts": counts,
        "epochs": epochs,
        "train_loss": metrics.get("train_loss"),
        "train_runtime_s": metrics.get("train_runtime"),
        "peak_vram_gb": round(peak_gb, 2),
        "gpu": torch.cuda.get_device_name(0),
    }


@app.local_entrypoint()
def main(epochs: int = 3, chat_rows: int = 300, out: str = "data/run.json"):
    result = run.remote(epochs, chat_rows)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(f"wrote {out}")
