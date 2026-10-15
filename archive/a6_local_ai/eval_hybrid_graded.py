"""Greedy graded-10 with hybrid best adapter; compare to baseline + public_only."""

from __future__ import annotations

import json
import logging
import os
import shutil
import warnings
from datetime import UTC, datetime
from pathlib import Path

import torch
from evaluate import load as load_metric
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "assignment6"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER = DATA / "train_hybrid" / "best"
OUT = DATA / "after_hybrid_train.json"
REPORT = DATA / "after_hybrid_train_report.md"
SNAP = DATA / "adapters" / "hybrid_v1"
MAX_NEW_TOKENS = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRIORITY = {1, 2, 3, 5, 9}

HF_TOKEN = os.environ.get("HF_TOKEN")
token_path = Path.home() / ".cache" / "huggingface" / "token"
if HF_TOKEN is None and token_path.is_file():
    HF_TOKEN = token_path.read_text(encoding="utf-8").strip() or None

gold = json.loads((DATA / "gold_answers.json").read_text())
baseline = json.loads((DATA / "baseline_before_training.json").read_text())
public = json.loads(
    (DATA / "adapters" / "public_only" / "after_public_train.json").read_text()
)


def norm(text: str) -> str:
    return " ".join(str(text).lower().split())


def has_phrase(text: str, phrase: str) -> bool:
    return norm(phrase) in norm(text)


def keyword_pass(prediction: str, item: dict) -> bool:
    if any(has_phrase(prediction, bad) for bad in item.get("reject_if") or []):
        return False
    keywords = item.get("keywords") or []
    if not keywords:
        return False
    hits = [kw for kw in keywords if has_phrase(prediction, kw)]
    if item.get("match") == "all":
        return len(hits) == len(keywords)
    return bool(hits)


print(f"Device={DEVICE} adapter={ADAPTER}")
assert ADAPTER.is_dir(), ADAPTER
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    attn_implementation="sdpa",
)
model = PeftModel.from_pretrained(model, ADAPTER)
model.to(DEVICE)
model.eval()


def ask(question: str) -> str:
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


examples = []
for i, item in enumerate(gold, 1):
    ans = ask(item["question"])
    examples.append({"question": item["question"], "answer": ans})
    print(f"=== Q{i} ===\n{ans[:200]}\n")

payload = {
    "model": MODEL_ID,
    "adapter": str(ADAPTER.relative_to(REPO)),
    "device": DEVICE,
    "max_new_tokens": MAX_NEW_TOKENS,
    "generation": "greedy",
    "created_utc": datetime.now(UTC).isoformat(),
    "examples": examples,
}
OUT.write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)


def score(examples_list):
    rows = []
    for pred, item in zip(examples_list, gold, strict=True):
        rows.append(
            {
                "id": item["id"],
                "pass": keyword_pass(pred["answer"], item),
                "expected": item["expected"],
                "model": pred["answer"],
                "question": item["question"],
            }
        )
    preds = [r["model"] for r in rows]
    refs = [r["expected"] for r in rows]
    warnings.filterwarnings("ignore")
    import transformers

    transformers.logging.set_verbosity_error()
    logging.getLogger("transformers").setLevel(logging.ERROR)
    rouge = load_metric("rouge").compute(
        predictions=preds,
        references=refs,
        rouge_types=["rouge1", "rougeL"],
        use_stemmer=False,
    )
    bert = load_metric("bertscore").compute(
        predictions=preds,
        references=refs,
        lang="en",
        model_type="roberta-base",
        idf=False,
        rescale_with_baseline=False,
        device="cpu",
    )
    soft = {
        "rouge1": round(float(rouge["rouge1"]), 3),
        "rougeL": round(float(rouge["rougeL"]), 3),
        "bertscore_f1": round(float(sum(bert["f1"]) / len(bert["f1"])), 3),
    }
    return rows, soft


hy_rows, hy_soft = score(examples)
base_rows, base_soft = score(baseline["examples"])
pub_rows, pub_soft = score(public["examples"])
hy_pass = sum(r["pass"] for r in hy_rows)
base_pass = sum(r["pass"] for r in base_rows)
pub_pass = sum(r["pass"] for r in pub_rows)
hy_pri = sum(r["pass"] for r in hy_rows if r["id"] in PRIORITY)
pub_pri = sum(r["pass"] for r in pub_rows if r["id"] in PRIORITY)

lines = [
    "# After hybrid-train score report",
    "",
    f"- Adapter: `{ADAPTER.relative_to(REPO)}`",
    f"- Keyword: baseline {base_pass}/10 → public_only {pub_pass}/10 → **hybrid {hy_pass}/10**",
    f"- Priority Q1/2/3/5/9: public_only {pub_pri}/5 → **hybrid {hy_pri}/5**",
    f"- Soft hybrid: ROUGE-1={hy_soft['rouge1']}, ROUGE-L={hy_soft['rougeL']}, BERTScore={hy_soft['bertscore_f1']}",
    f"- Soft public_only: ROUGE-1={pub_soft['rouge1']}, ROUGE-L={pub_soft['rougeL']}, BERTScore={pub_soft['bertscore_f1']}",
    "",
    "| ID | Baseline | Public | Hybrid | Expected |",
    "| ---: | :---: | :---: | :---: | --- |",
]
for b, p, h in zip(base_rows, pub_rows, hy_rows, strict=True):
    lines.append(
        f"| {h['id']} | {'✓' if b['pass'] else '✗'} | {'✓' if p['pass'] else '✗'} | "
        f"{'✓' if h['pass'] else '✗'} | {h['expected'].replace('|', '/')} |"
    )
lines += ["", "## Per-question (hybrid)", ""]
for h in hy_rows:
    lines += [
        f"### Q{h['id']}. {h['question']}",
        "",
        f"**Expected:** {h['expected']}",
        "",
        f"**Model:** {h['model']}",
        "",
        f"- Keywords: {'pass' if h['pass'] else 'fail'}",
        "",
    ]
REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

# Snapshot adapter if priority bar met or always keep hybrid_v1 for record
if SNAP.exists():
    shutil.rmtree(SNAP)
SNAP.mkdir(parents=True)
shutil.copytree(ADAPTER, SNAP, dirs_exist_ok=True)
(SNAP / "after_hybrid_train.json").write_text(
    OUT.read_text(encoding="utf-8"), encoding="utf-8"
)
(SNAP / "after_hybrid_train_report.md").write_text(
    REPORT.read_text(encoding="utf-8"), encoding="utf-8"
)
(SNAP / "README.md").write_text(
    "\n".join(
        [
            "# Hybrid v1 LoRA adapter",
            "",
            f"- Source: `{ADAPTER}`",
            f"- Graded keyword: {hy_pass}/10 (priority {hy_pri}/5)",
            f"- vs public_only: {pub_pass}/10 (priority {pub_pri}/5)",
            f"- Created: {datetime.now(UTC).isoformat()}",
            "",
        ]
    ),
    encoding="utf-8",
)

decision = {
    "baseline_keyword_pass": base_pass,
    "public_keyword_pass": pub_pass,
    "hybrid_keyword_pass": hy_pass,
    "public_priority_pass": pub_pri,
    "hybrid_priority_pass": hy_pri,
    "success_bar_priority_2_of_5": hy_pri >= 2,
    "soft_hybrid": hy_soft,
    "soft_public": pub_soft,
}
(DATA / "hybrid_train_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
print(json.dumps(decision, indent=2))
print(f"Wrote {OUT}\nWrote {REPORT}\nSnapshot {SNAP}")
