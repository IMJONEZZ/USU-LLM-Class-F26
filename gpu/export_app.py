"""Rebuilds the assignment 6 model and saves it this time.

The model I fine tuned in assignment 6 doesn't exist any more. It trained on a
rented GPU, the container shut down, and the adapter went with it. Same story
for assignment 5. Neither trainer ever wrote anything to disk.

So this retrains it from the same seeded data and writes three things to a
volume that outlives the run:

  * the LoRA adapter, which is small and is the thing I should have kept
  * a merged GGUF at q4_k_m, which is what the server in `gpu/serve_app.py`
    loads
  * a manifest with the answers it gave, so I can tell whether the rebuild
    matches what assignment 6 reported before anything gets quantized

    modal run gpu/export_app.py
"""

import json

import modal

GPU_TYPE = "T4"
CACHE_DIR = "/cache"
MODEL_DIR = "/models"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "cmake", "git", "curl", "libcurl4-openssl-dev")
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
        # Unsloth shells out to llama.cpp to build the GGUF and wants these.
        # Left unpinned on purpose: unsloth caps protobuf below 4.0 and pinning
        # a newer one makes the whole install unresolvable.
        "gguf",
        "sentencepiece",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": CACHE_DIR})
    .add_local_python_source("src")
    .add_local_file("data/paraphrases.json", "/root/paraphrases.json")
)

app = modal.App("hw7-export", image=image)
hf_cache = modal.Volume.from_name("hw6-hf-cache", create_if_missing=True)
models = modal.Volume.from_name("hw7-models", create_if_missing=True)


@app.function(
    gpu=GPU_TYPE,
    timeout=7200,
    volumes={CACHE_DIR: hf_cache, MODEL_DIR: models},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def export(epochs: int = 3, chat_rows: int = 800):
    import os
    import shutil

    from src.dataset import (
        HOMEWORK_QUESTIONS,
        build_dataset,
        load_chat_rows,
        to_messages,
    )
    from src.trainer import (
        GGUF_QUANTIZATION,
        answer,
        attach_lora,
        finetune,
        load_base_model,
        save_adapter,
        save_gguf,
        save_merged,
    )

    with open("/root/paraphrases.json", encoding="utf-8") as handle:
        paraphrases = json.load(handle)

    rows = build_dataset(
        chat_rows=load_chat_rows(limit=chat_rows), paraphrases=paraphrases
    )
    print(f"training rows: {len(rows)}")

    model, tokenizer = load_base_model()
    model = attach_lora(model)
    stats = finetune(model, tokenizer, to_messages(rows), epochs=epochs)

    # Ask before saving. If the rebuild didn't land the same way assignment 6
    # did, I want to know that now and not after it's been quantized.
    answers = answer(model, tokenizer, HOMEWORK_QUESTIONS)
    print("\n=== rebuilt model, before any quantizing ===")
    for i, (question, reply) in enumerate(zip(HOMEWORK_QUESTIONS, answers), 1):
        print(f"\nQ{i}: {question}\nA: {reply}")

    adapter_dir = f"{MODEL_DIR}/adapter"
    gguf_dir = f"{MODEL_DIR}/gguf"
    for path in (adapter_dir, gguf_dir):
        shutil.rmtree(path, ignore_errors=True)

    save_adapter(model, tokenizer, adapter_dir)
    print(f"\nadapter written to {adapter_dir}")

    gguf_file, merged_dir = None, None
    try:
        save_gguf(model, tokenizer, gguf_dir)
        # Unsloth leaves the F16 intermediate next to the quantized file, so
        # picking the first .gguf off the directory listing is a coin flip.
        # The first time I ran this it handed me the 2.5GB F16 and I nearly
        # served that instead of the 800MB one I asked for.
        found = sorted(f for f in os.listdir(gguf_dir) if f.endswith(".gguf"))
        wanted = [f for f in found if GGUF_QUANTIZATION.upper() in f.upper()]
        gguf_file = f"{gguf_dir}/{(wanted or found)[0]}" if found else None
        for name in found:
            size = os.path.getsize(f"{gguf_dir}/{name}") / 1024**3
            print(
                f"  {name}  {size:.2f} GB{'   <- serving this' if f'{gguf_dir}/{name}' == gguf_file else ''}"
            )
    except Exception as exc:  # noqa: BLE001 - the fallback is the whole point
        print(f"\nGGUF export failed: {type(exc).__name__}: {exc}")
        print("falling back to merged 16 bit so the weights still get out")
        merged_dir = f"{MODEL_DIR}/merged"
        shutil.rmtree(merged_dir, ignore_errors=True)
        save_merged(model, tokenizer, merged_dir)

    manifest = {
        "questions": HOMEWORK_QUESTIONS,
        "answers": answers,
        "adapter_dir": adapter_dir,
        "gguf_file": gguf_file,
        "merged_dir": merged_dir,
        "epochs": epochs,
        "rows": len(rows),
        "train_loss": dict(stats.metrics).get("train_loss"),
    }
    with open(f"{MODEL_DIR}/manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    models.commit()
    print("\nvolume committed")
    return manifest


@app.local_entrypoint()
def main(epochs: int = 3, chat_rows: int = 800, out: str = "data/export.json"):
    manifest = export.remote(epochs, chat_rows)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(f"wrote {out}")
