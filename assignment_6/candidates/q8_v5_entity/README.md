# Separate Q8 candidate from protected run12

> **Current state, 2026-10-14:** Run16 is FINAL, student-accepted5/10(Q2/Q5/Q6/Q7/Q8), Q1 regression accepted. Active notebook/results now use run16. Run12 raw evidence/former submission copies are archived unchanged. Start with A6_START_HERE.md, A6_FINAL_SELECTION.json and latest Exact resume in A6_HANDOVER.md. Earlier pending/run12-selected statements below are historical and superseded. No further model experiment authorized.

## Historical preparation and experiment record

Prepared on 2026-10-14 after Pallavi said: "Okay, let's do it based on all this research plan."
This implements candidate preparation only; training remains a separate decision.

- Exact archived v5: 522 rows, 426 train / 96 dev.
- 31 existing Q8 training references rewritten: 22 generic fiction + 9 existing non-Greek machines/software.
- Each rewritten row's lexical diagnostic metadata now matches the reference.
- Vlad alone changes skill from fiction_vs_history to historical_person.
- All prompts, IDs, families, archive order, exact split order and membership remain unchanged.
- 491 archive rows and all 96 development rows remain unchanged.
- Golden-maidens row and seven other Q8 training rows remain unchanged.
- Protected run12 notebook/results/adapter/guide stay unchanged; later working v8 stays intact.
- No added prompts, Talos examples, new close analogues, weighting changes or settings sweep.

Read review.md for all exact before/after references and sources.
Read training_audit.json for the disposition of all 39 training rows.
Read development_spec.json and cached_run12_review.json for the frozen manual diagnostic.
The semantic review is an assistant diagnostic, not an assignment score or automatic grader.
It is not wired into checkpoint selection. Decide the selection policy before a run.

Reproduce from the bundled exact baseline input copies:

    python3 -B assignment_6/candidates/q8_v5_entity/build_candidate.py
    python3 -B assignment_6/candidates/q8_v5_entity/build_candidate.py --check

No import from the later v8 generator is used. The baseline input copies are hashed against
the protected archive and actual run12 manifest, with original paths recorded in replacements.json.
All 14 protected hashes are additionally checked when the local snapshot is available.

The existing Modal entrypoint still loads v8 and recomputes a stratified split.
DO NOT run it unchanged for this candidate. A manifest-aware isolated entrypoint,
actual trainer/evaluator provenance, tokenizer length audit and reviewed selection policy
are required before any separately authorized model run.
Proposed future name: v5_q8_entity, fresh pinned base + new LoRA, original run12 settings.
data_manifest.json records the original config as history, not authorization to write to
its old output_dir. Future output must use a new prefix; never v5_controlled/.

Any candidate must earn Q1/Q5/Q6/Q7 and Q8 itself under the unchanged guide.
Related-fact/same-trick permission remains unresolved. No instructor approval is claimed.

## Authorized isolated run (2026-10-14)

Pallavi authorized starting one controlled run. run_recipe.json is the active
authorization/settings record; preparation-only flags in data_manifest.json describe
the historical prepared snapshot. The isolated modal_q8_job.py now loads that
manifest exactly. selection_policy.json fixes the ten-group macro, replacing only
Q8 with the frozen manual semantic complete/8 diagnostic. Automatic trainer
selection is provisional; finalize exports the development-selected checkpoint
before answers. Token lengths and runtime checks are integrated before/after
training. Current v8 and protected run12 remain unchanged.53tests/Ruff pass.

## Completed outcome

The one authorized run is complete. See A6_Q8_V5_ENTITY_RESULTS.md and
A6_RUNS/run16_v5_q8_entity/. Q8 and Q2 improved, but Q1 regressed; assistant
provisional5/10(Q2,Q5,Q6,Q7,Q8), student scoring pending. The required retained
four+Q8 condition fails. Run12 remains selected and protected. No further run
is authorized; existing-prefix/repeated-inference guards remain active.


## Current offline regeneration

From repo-a6 use python3 -B assignment_6/final_submission.py to verify final data/notebook. The wrapper supplies current protection hashes without changing this frozen builder. The direct historical commands above used obsolete active run12 hashes. Final data:assignment_6/results/training_dataset.jsonl. Raw automatic best108 is provisional history; final checkpoint162. Current55tests/94.61%coverage/Ruff pass. No further run authorized.
