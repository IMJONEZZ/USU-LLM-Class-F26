"""Rewords the trick questions with a bigger model, through distilabel.

My templates only give three phrasings per family, and the questions I'm
actually graded on are worded nothing like mine. So this asks Llama 3.2 3B to
say the same question a different way, four times each.

The answers never come from the model. They were worked out from the slots in
``src/dataset.py``, and a reworded question only gets kept if it still has the
same numbers in it. A paraphrase that turned 600 people into 60 would leave me
with a row whose answer is quietly wrong, and that's the failure I'd never
notice by reading the data.

I started out driving this with distilabel's `Pipeline` and it died inside the
library's own logging setup once it forked a worker, so this calls
`TransformersLLM` directly instead. Same library, one less layer.

    modal run gpu/paraphrase_app.py
"""

import json
import logging

import modal

GPU_TYPE = "T4"
CACHE_DIR = "/cache"
PARAPHRASER = "meta-llama/Llama-3.2-3B-Instruct"
COPIES = 4
CHUNK = 24

PROMPT = """Reword the question below so it asks exactly the same thing in different words.

Rules:
- Keep every number exactly as it is.
- Do not answer the question.
- Reply with the reworded question only, nothing else.

Question: {question}"""

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.49.0",
        "accelerate==1.4.0",
        "distilabel==1.5.3",
        # distilabel picks the CUDA device through pynvml and won't load without it
        "pynvml==11.5.3",
        "huggingface_hub[hf_transfer]==0.29.1",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": CACHE_DIR})
    .add_local_python_source("src")
)

app = modal.App("hw6-paraphrase", image=image)
hf_cache = modal.Volume.from_name("hw6-hf-cache", create_if_missing=True)


def _texts(result):
    """Pull the generated strings out of whatever shape distilabel returned."""
    if isinstance(result, dict):
        return result.get("generations") or []
    if isinstance(result, list):
        return [r if isinstance(r, str) else str(r) for r in result]
    return []


@app.function(
    gpu=GPU_TYPE,
    timeout=3600,
    volumes={CACHE_DIR: hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def paraphrase(copies: int = COPIES):
    from distilabel.models.llms import TransformersLLM

    from src.dataset import build_trick_questions, paraphrase_rows

    originals = build_trick_questions()
    print(f"{len(originals)} questions, {copies} rewordings each")

    llm = TransformersLLM(
        model=PARAPHRASER,
        torch_dtype="float16",
        device="cuda",
    )
    # distilabel normally hands the LLM a logger when a pipeline Step loads it.
    # Calling load() myself skips that, and the CUDA placement code logs
    # unconditionally, so without this it dies on a None logger.
    llm._logger = logging.getLogger("distilabel.paraphrase")
    llm.load()

    originals_expanded, raw_generations = [], []
    for start in range(0, len(originals), CHUNK):
        batch = originals[start : start + CHUNK]
        conversations = [
            [{"role": "user", "content": PROMPT.format(question=row["question"])}]
            for row in batch
        ]
        results = llm.generate_outputs(
            inputs=conversations,
            num_generations=copies,
            temperature=0.9,
            top_p=0.95,
            max_new_tokens=90,
            do_sample=True,
        )
        for row, result in zip(batch, results):
            for text in _texts(result):
                originals_expanded.append(row)
                raw_generations.append(text)
        print(f"  {min(start + CHUNK, len(originals))}/{len(originals)} questions done")

    kept = paraphrase_rows(originals_expanded, raw_generations)

    # The same wording turns up more than once across four samples.
    seen, deduped = set(), []
    for row in kept:
        key = row["question"].strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(row)

    print(f"kept {len(kept)} of {len(raw_generations)}, {len(deduped)} after dedupe")
    return deduped


@app.local_entrypoint()
def main(copies: int = COPIES, out: str = "data/paraphrases.json"):
    rows = paraphrase.remote(copies)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
    print(f"wrote {len(rows)} paraphrases to {out}")
