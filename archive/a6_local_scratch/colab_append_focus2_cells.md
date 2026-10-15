# Focus2 append cells (paste under your current Colab notebook)

No runtime restart. Upload **once** to Drive:

1. `MyDrive/usu-a6/a6_focus2_data.tgz` — new skill data only
2. Keep `MyDrive/usu-a6/a6_focus_results.tgz` — previous focus adapter (init)

Session must already have: repo at `/content/USU-LLM-Class-F26`, deps installed, HF logged in, Drive mounted.

Data: ~238 train / 26 dev — Christmas, sphinx (+man/human), arthritis, parallel→3, feathers, light keepers.

---

## CELL A — unpack data + previous adapter from Drive

```python
# A) Unpack focus2 data + stage init adapter from a6_focus_results.tgz
import os
import shutil
import tarfile
from pathlib import Path

from google.colab import drive

drive.mount("/content/drive")

REPO = Path("/content/USU-LLM-Class-F26")
assert REPO.is_dir(), "Repo missing — run earlier setup cells first"
os.chdir(REPO)

DATA_TGZ = Path("/content/drive/MyDrive/usu-a6/a6_focus2_data.tgz")
assert DATA_TGZ.is_file(), f"Upload a6_focus2_data.tgz to MyDrive/usu-a6/ first ({DATA_TGZ})"

PACK = Path("/content/a6_focus2_data")
if PACK.exists():
    shutil.rmtree(PACK)
PACK.mkdir(parents=True)
with tarfile.open(DATA_TGZ, "r:gz") as tar:
    tar.extractall(PACK)
root = PACK
kids = [p for p in PACK.iterdir()]
if len(kids) == 1 and kids[0].is_dir():
    root = kids[0]

QA = Path("data/assignment6/qa")
QA.mkdir(parents=True, exist_ok=True)
shutil.copy(root / "qa" / "focus2_train.jsonl", QA / "focus2_train.jsonl")
shutil.copy(root / "qa" / "focus2_dev.jsonl", QA / "focus2_dev.jsonl")
if (root / "gold_answers.json").is_file():
    shutil.copy(root / "gold_answers.json", Path("data/assignment6/gold_answers.json"))
print("data lines:", sum(1 for _ in open(QA / "focus2_train.jsonl")), "train,", sum(1 for _ in open(QA / "focus2_dev.jsonl")), "dev")

# Prefer in-session train_focus/best; else restore from Drive results
ADAPTER = Path("data/assignment6/train_focus/best")
if not (ADAPTER / "adapter_config.json").is_file():
    results = Path("/content/drive/MyDrive/usu-a6/a6_focus_results.tgz")
    assert results.is_file(), "Need train_focus/best in session OR a6_focus_results.tgz on Drive"
    R = Path("/content/a6_focus_results_unpack")
    if R.exists():
        shutil.rmtree(R)
    R.mkdir(parents=True)
    with tarfile.open(results, "r:gz") as tar:
        tar.extractall(R)
    hits = list(R.rglob("adapter_config.json"))
    prefer = [p for p in hits if "best" in str(p) or "adapter_best" in str(p) or "train_focus" in str(p)]
    src = (prefer or hits)[0].parent
    ADAPTER.parent.mkdir(parents=True, exist_ok=True)
    if ADAPTER.exists():
        shutil.rmtree(ADAPTER)
    shutil.copytree(src, ADAPTER)
    print("restored adapter from Drive:", src)
else:
    print("using existing", ADAPTER)

dst = Path("data/assignment6/adapters/focus2_start")
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(ADAPTER, dst)
print("init-adapter staged:", dst)
```

---

## CELL B — validate + prepare-qa

```python
# B) Validate + prepare-qa
import json
import os
import shutil
import subprocess
from pathlib import Path

from src.trainer import find_graded_leakage, validate_qa_record

os.chdir("/content/USU-LLM-Class-F26")
train_path = Path("data/assignment6/qa/focus2_train.jsonl")
dev_path = Path("data/assignment6/qa/focus2_dev.jsonl")


def load_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


train_rows, dev_rows = load_jsonl(train_path), load_jsonl(dev_path)
for row in train_rows + dev_rows:
    validate_qa_record(row)
    assert row["review_status"] == "approved"
leak = find_graded_leakage(train_rows + dev_rows)
assert not leak, leak[:3]
print(f"OK train={len(train_rows)} dev={len(dev_rows)}")

out_prep = Path("data/assignment6/prepared_focus2")
if out_prep.exists():
    shutil.rmtree(out_prep)
subprocess.check_call(
    [
        "python",
        "-m",
        "src.trainer",
        "prepare-qa",
        "--train",
        str(train_path),
        "--dev",
        str(dev_path),
        "--output",
        str(out_prep),
    ]
)
print("prepared_focus2 OK")
```

---

## CELL C — knobs

```python
# C) A100 knobs for small dense focus2
BATCH_SIZE = 256
ACCUMULATION = 1
EPOCHS = 8
FEASIBILITY_STEPS = 2
print("effective_batch", BATCH_SIZE * ACCUMULATION, "epochs", EPOCHS)
```

---

## CELL D — train from previous focus adapter

```python
# D) Feasibility + train focus2 (continues older adapter)
import os
import shutil
import subprocess
from pathlib import Path

os.chdir("/content/USU-LLM-Class-F26")
for d in ["data/assignment6/feasibility_focus2", "data/assignment6/train_focus2"]:
    p = Path(d)
    if p.exists():
        shutil.rmtree(p)

common = [
    "--prepared",
    "data/assignment6/prepared_focus2",
    "--init-adapter",
    "data/assignment6/adapters/focus2_start",
    "--batch-size",
    str(BATCH_SIZE),
    "--accumulation",
    str(ACCUMULATION),
    "--learning-rate",
    "2e-4",
    "--rank",
    "4",
    "--alpha",
    "8",
    "--targets",
    "q_proj,v_proj",
]
help_txt = subprocess.getoutput("python -m src.trainer train -h")
if "--no-gradient-checkpointing" in help_txt or "--gradient-checkpointing" in help_txt:
    common.append("--no-gradient-checkpointing")

subprocess.check_call(
    [
        "python",
        "-m",
        "src.trainer",
        "feasibility",
        *common,
        "--output",
        "data/assignment6/feasibility_focus2",
        "--steps",
        str(FEASIBILITY_STEPS),
    ]
)
subprocess.check_call(
    [
        "python",
        "-m",
        "src.trainer",
        "train",
        *common,
        "--feasibility",
        "data/assignment6/feasibility_focus2",
        "--output",
        "data/assignment6/train_focus2",
        "--epochs",
        str(EPOCHS),
    ]
)
assert Path("data/assignment6/train_focus2/success.json").is_file()
print("TRAIN_OK focus2")
print(subprocess.getoutput("nvidia-smi"))
```

---

## CELL E — graded answers

```python
# E) Graded-10 with train_focus2/best
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.trainer import GRADED_QUESTIONS

os.chdir("/content/USU-LLM-Class-F26")
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER = Path("data/assignment6/train_focus2/best")
OUT = Path("data/assignment6/after_focus2_train.json")
DEVICE = "cuda"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, attn_implementation="sdpa"
)
model = PeftModel.from_pretrained(base, ADAPTER)
model.to(DEVICE)
model.eval()


def ask(question: str) -> str:
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


examples = []
for q in GRADED_QUESTIONS:
    ans = ask(q)
    examples.append({"question": q, "answer": ans})
    print("Q:", q)
    print("A:", ans)
    print("---")

payload = {
    "model": MODEL_ID,
    "adapter": str(ADAPTER),
    "device": DEVICE,
    "max_new_tokens": 128,
    "generation": "greedy",
    "created_utc": datetime.now(UTC).isoformat(),
    "examples": examples,
}
OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
print("Wrote", OUT)
```

---

## CELL F — keyword grade + save to Drive

```python
# F) Keyword grade + save a6_focus2_results.tgz
import json
import os
import shutil
import tarfile
from pathlib import Path

os.chdir("/content/USU-LLM-Class-F26")
DATA = Path("data/assignment6")
gold = json.loads((DATA / "gold_answers.json").read_text())
pred = json.loads((DATA / "after_focus2_train.json").read_text())


def norm(text: str) -> str:
    return " ".join(str(text).lower().split())


def keyword_pass(prediction: str, item: dict) -> bool:
    if any(norm(bad) in norm(prediction) for bad in item.get("reject_if") or []):
        return False
    keywords = item.get("keywords") or []
    hits = [kw for kw in keywords if norm(kw) in norm(prediction)]
    if item.get("match") == "all":
        return len(hits) == len(keywords)
    return bool(hits)


passed = 0
lines = ["# Focus2 graded keyword report", ""]
for item, ex in zip(gold, pred["examples"], strict=True):
    ok = keyword_pass(ex["answer"], item)
    passed += int(ok)
    mark = "PASS" if ok else "FAIL"
    print(f"Q{item['id']}: {mark} | {ex['answer'][:120]}")
    lines.append(f"- Q{item['id']}: **{mark}** — {ex['answer']}")
lines += ["", f"**Keyword score: {passed}/{len(gold)}**", ""]
report = DATA / "after_focus2_train_report.md"
report.write_text("\n".join(lines), encoding="utf-8")
print(f"Keyword {passed}/{len(gold)}")

out_dir = Path("/content/drive/MyDrive/usu-a6/results_focus2")
if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True)
shutil.copytree(DATA / "train_focus2" / "best", out_dir / "adapter_best")
shutil.copy(DATA / "after_focus2_train.json", out_dir / "after_focus2_train.json")
shutil.copy(report, out_dir / "after_focus2_train_report.md")
shutil.copy(DATA / "train_focus2" / "training.json", out_dir / "training.json")
shutil.copy(DATA / "train_focus2" / "success.json", out_dir / "success.json")

results_tgz = Path("/content/drive/MyDrive/usu-a6/a6_focus2_results.tgz")
with tarfile.open(results_tgz, "w:gz") as tar:
    tar.add(out_dir, arcname="results_focus2")
print("Saved", out_dir)
print("Saved", results_tgz)
print("DONE")
```
