# Public-only LoRA baseline (frozen record)

Snapshot of the Assignment 6 run trained only on approved public QA
(BrainTeaser, FalseQA, StrategyQA, TruthfulQA). Do not overwrite this
directory; later hybrid runs use separate paths under `data/assignment6/`.

## Identity

| Field | Value |
| --- | --- |
| Base model | `meta-llama/Llama-3.2-1B-Instruct` |
| Revision | `9213176726f574b556790deb65791e0c5aa438b6` |
| Adapter path | `data/assignment6/adapters/public_only/` |
| Original run | `data/assignment6/train/best` (epoch 3) |
| Snapshot date (UTC) | 2026-10-12 |

**Base Instruct Hub weights were not modified.** Only this LoRA adapter was trained and saved.

## LoRA / train config

- Rank `r=4`, alpha `8`, targets `q_proj,v_proj`
- Batch size 1, accumulation 8, LR `2e-4`, epochs 3, seed 0
- Best-on-dev selection: epoch **3**, `checkpoint_dev_loss` **1.6727**
- Prepared train/dev sizes: 7182 / 798

## Graded-10 vs no-adapter baseline

| Metric | Baseline (no adapter) | This adapter |
| --- | ---: | ---: |
| Keyword pass | 0/10 | **2/10** (Q8, Q10) |
| ROUGE-1 / ROUGE-L | 0.098 / 0.075 | 0.144 / 0.119 |
| BERTScore F1 | 0.822 | 0.855 |
| Priority stems Q1/2/3/5/9 | 0/5 | 0/5 |

Artifacts in this folder (and `record/`):

- `adapter_model.safetensors`, `adapter_config.json`, `run.json`
- `after_public_train.json` — greedy answers with this adapter
- `after_public_train_report.md`, `public_train_status.md`, `public_train_decision.json`
- `training.json` — full epoch history

Auto-generated PEFT notes (if present): `README.peft.md`.

## Load hint

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", torch_dtype="float16")
model = PeftModel.from_pretrained(model, "data/assignment6/adapters/public_only")
```
