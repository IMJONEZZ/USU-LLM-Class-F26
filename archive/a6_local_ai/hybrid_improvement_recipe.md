# Hybrid improvement recipe (next run)

Public-only record stays frozen at [`adapters/public_only/`](adapters/public_only/). This document is the execute checklist for a skill-targeted hybrid retrain. Do not overwrite public `qa/train.jsonl`, `qa/dev.jsonl`, `prepared/`, `train/`, or `adapters/public_only/`.

## 1. ChatGPT skill targets (paste prompts; no agent API)

Generate **about 50–150** short instruction/response pairs **per skill**, paraphrased — never the graded stems:

| Skill id | Category string to use | Target stems helped |
| --- | --- | --- |
| `false_scaling` | `false_scaling` | Q1 |
| `currency_mass_ambiguity` | `currency_mass_ambiguity` | Q2 |
| `situational_riddle` | `situational_riddle` | Q3 |
| `absurd_premise_safety` | `absurd_premise_safety` | Q5 |
| `letter_counting` | `letter_counting` | Q9 |
| `harmless_misconception` | `harmless_misconception` | Q4 |
| `constraint_joke` | `constraint_joke` | Q6 |
| `current_fact` | `current_fact` | Q7 (optional thin slice) |

Answer style: short, direct, same tone as gold (`gold_answers.json`) — not long essays.

### JSONL record shape (required fields)

Each line must pass `validate_qa_record` in `src/trainer.py`:

```json
{
  "example_id": "synth:false_scaling:0001",
  "instruction": "...",
  "response": "...",
  "category": "false_scaling",
  "source": "chatgpt:skill:false_scaling",
  "review_status": "pending"
}
```

Suggested landing files (pending until leakage review):

- `data/assignment6/qa/pending_synth.jsonl` — all new synth rows
- After approve: merge into hybrid paths below (do not mutate public approved files)

### Leakage gate (mandatory before approve)

```bash
uv run python -m src.build_qa_dataset inspect  # or the leakage scan path you already use
# Ensure no instruction normalizes to / near-duplicates GRADED_QUESTIONS
```

`prepare-qa` will also exclude graded-stem collisions. Prefer catching them while rows are still `pending`.

## 2. Hybrid approved paths

Keep public approved files untouched:

- Public: `qa/train.jsonl`, `qa/dev.jsonl`
- Hybrid (new):
  - `qa/hybrid_train.jsonl` = public train + approved synth train
  - `qa/hybrid_dev.jsonl` = public dev + small approved synth dev slice (for monitoring)

Approve synth by setting `review_status` to `"approved"` only after manual/leak review.

## 3. Fresh prepare / feasibility / train dirs

Trainer refuses existing output directories — use new names:

```bash
# prepare
uv run python -m src.trainer prepare-qa \
  --train data/assignment6/qa/hybrid_train.jsonl \
  --dev data/assignment6/qa/hybrid_dev.jsonl \
  --output data/assignment6/prepared_hybrid

# Prefer .venv-training on this host (same as public-only run)
.venv-training/bin/python -m src.trainer feasibility \
  --prepared data/assignment6/prepared_hybrid \
  --output data/assignment6/feasibility_hybrid

.venv-training/bin/python -m src.trainer train \
  --prepared data/assignment6/prepared_hybrid \
  --feasibility data/assignment6/feasibility_hybrid \
  --output data/assignment6/train_hybrid
```

Same LoRA recipe as public-only: r=4, alpha=8, q/v, batch 1, accum 8, LR 2e-4, 3 epochs, best-on-dev.

Optional durable snapshot after a good hybrid run:

- `data/assignment6/adapters/hybrid_v1/` ← copy of `train_hybrid/best`

## 4. Compare graded-10

Generate greedy answers with `train_hybrid/best` (same protocol as `after_public_train.json`).

Compare against **both**:

1. No-adapter baseline: `baseline_before_training.json`
2. Frozen public-only: `adapters/public_only/after_public_train.json`

Save e.g. `data/assignment6/after_hybrid_train.json` + report.

### Success bar

- Keyword gains on priority set **Q1/2/3/5/9**: at least **2–3 of 5**
- Do not badly regress Q8/Q10 (still pass or soft-close)
- Soft ROUGE/BERTScore secondary to priority keyword gains

## 5. Out of scope reminders

- No commit/push/PR / invented Action Plan text unless you ask later
- No overwriting Hub Instruct weights
- No graded stems in train
- Defer `answers.ipynb` until hybrid beats public-only on priorities
