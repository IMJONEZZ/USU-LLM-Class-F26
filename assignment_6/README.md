# Assignment 6 final model and submission

## Start here (for a first-time reviewer)

1. **See the result:** open `answers.ipynb`. It asks the ten assignment questions and prints the fine-tuned model's actual, unedited answers (all from one model).
2. **Score:** 6/10 if Q9's correct count ("3 Ls") earns credit, 5/10 if its wrong optional spelling disqualifies it. Assignment credit for answers is capped at five. These are the student's judgments, not an instructor grade.
3. **Training code and tests:** `src/trainer.py`, `src/a6_evaluation.py`, `tests/`. The GPU job is `assignment_6/modal_job.py`.
4. **Exact data and results:** `assignment_6/results/` (training data, train/dev split, training history, raw answers).
5. **Older files:** names such as run16, v5 or v8 are internal experiment labels; `targeted_examples.jsonl` is an earlier version, not the final training data.

---

Pallavi selected run16_v5_q8_entity as final. She accepts **6/10** under the guide's count criterion: **Q2, Q5, Q6, Q7, Q8, Q9**. Q9 states the correct count, 3 Ls, but adds an incorrect optional spelling. If a grader treats that extra error as disqualifying, the conservative count is **5/10** (Q2, Q5, Q6, Q7, Q8). The assignment caps answer credit at five points. Q1, Q3, Q4 and Q10 remain incorrect.
She accepted any five correct answers as sufficient, superseding the earlier
four-retained-plus-Q8 requirement. Q1's regression remains visible.
All ten outputs come from one model; no outputs were edited or combined.

## Final files

- answers.ipynb: printed verbatim GPU-generated answers, final checkpoint and score.
- assignment_6/results/selected_answers.json: exact run16 raw outputs.
- assignment_6/results/selection.json: final checkpoint162 plus student score.
- assignment_6/results/final_selection.json: authoritative student model/data decision.
- assignment_6/results/student_final_review.json: accepted scoring record.
- assignment_6/results/training_result.json: raw trainer history.
- assignment_6/results/data_manifest.json: actual426train/96dev rows/order.
- assignment_6/results/training_dataset.jsonl: canonical final522-row dataset.
- assignment_6/results/development_semantic_review.json: pre-assignment checkpoint review.
- assignment_6/candidates/q8_v5_entity/: frozen generator, inputs, policy and isolated runner.

**assignment_6/targeted_examples.jsonl and make_targeted_dataset.py are historical
v8 (500 rows), not the final model's training data.** They remain preserved with
their historical tests. The final dataset path above is explicit and verified.
Do not regenerate v8 and describe it as the final model's data.
The generic assignment_6/modal_job.py is also historical; run16 used the isolated
manifest-aware modal_q8_job.py.

## Model and provenance

Model: meta-llama/Llama-3.2-1B-Instruct.
Pinned base revision:9213176726f574b556790deb65791e0c5aa438b6.
Modal volume:a6-artifacts.
Final adapter:v5_q8_entity/adapter.
Final checkpoint:/artifacts/v5_q8_entity/adapter/checkpoint-162.
Adapter SHA256:2d2735af8c344dfa52c002e46a7a71574309c50297cd5c25ce8c46d020e445a5.
Dataset file SHA256:a25937c29fad3db9fe9943d455fbdc94bb9c9d497a66299bcb70e6a8b5d149be.
Actual model-data digest:a334344274170edd3315fa7deedf873becbb4aa53e2058fb21b0d0786ddfe554.

Fresh pinned base/new LoRA, LR1e-4,3passes,rank/effectivealpha16/16,
microbatch2 x accumulation4,seed42,length1024,linear,FP16/T4.
522 rows; exact426train/96dev run12 split.31Q8training references changed;
all other rows/prompts/splits preserved. No Talos examples were added.
Token max127; no truncation.162steps,420.25seconds,6.06GiB.
Real masking/update/provisional-adapter fresh-base reload checks passed.
Final checkpoint export matched its adapter hash before fresh-base inference.

Training_result.json's automatic best108/pending flag is preserved raw history.
Final policy replaced ONLY Q8's component with frozen manual semantic review,
keeping the other nine development components unchanged and equally weighted.
It selected pass3/checkpoint162 at2026-10-14T08:43:49.011247+00:00 BEFORE assignment inference.
selection.json and the selection embedded in selected_answers.json identify the
actual final model. Greedy answers,512new-token cap,no system prompt.

## Offline checks and regeneration

From repo-a6:

    python3 -B assignment_6/final_submission.py
    python3 -B assignment_6/final_submission.py --write-dataset
    uv run ruff check .
    uv run ruff format --check .
    uv run pytest

These check/reproduce the saved final dataset and notebook provenance; they do
not call a model. In the personal workspace, the current final protection profile
also checks the final files and original run12 archives. The original frozen
build_candidate.py CLI refers to the historical run12 selection snapshot; use
final_submission.py for the current selection, rather than rewriting old evidence.

## Saved history and remaining submission work

Run12 remains an unchanged accepted4/10 historical baseline. Its old notebook and
selected-result copies were archived before replacing the active submission.
Run16 adapter/tokenizer also has a local backup outside the worktree:
A6_RUNS/run16_v5_q8_entity/final_adapter/. Full run16 evidence and source/input
snapshots are outside the worktree under A6_RUNS/run16_v5_q8_entity/.

No more training or answer experiments are authorized. Final model selection is done.
Next: review the rubric and final diff, prepare/record the one-minute presentation,
obtain the student's own reflection/inline-note ideas, then commit/push and check
CI when those actions are authorized. Do not invent student reflection or revise
the first posted Action Plan comment. PR#62 is open; existing remote CI is green
for older pushed work, not proof of this still-uncommitted final state.

The correctness guide remains unchanged. Same-trick/related-fact interpretation
remains unresolved; no instructor approval or contact is implied.
For current cross-assistant context, read A6_START_HERE.md and the latest Exact
resume point in A6_HANDOVER.md in the parent workspace.
