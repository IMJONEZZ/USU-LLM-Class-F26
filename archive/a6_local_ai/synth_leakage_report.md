# Synth leakage review — pending_synth.jsonl

- File: `data/assignment6/qa/pending_synth.jsonl`
- Rows parsed/valid: **700**/700 expected
- Schema/status errors: **0**
- Duplicate example_ids: **0**
- Duplicate normalized instructions: **0**

## Category counts

| Category | N |
| --- | ---: |
| absurd_premise_safety | 100 |
| constraint_joke | 80 |
| currency_mass_ambiguity | 100 |
| current_fact | 40 |
| false_scaling | 100 |
| harmless_misconception | 80 |
| letter_counting | 100 |
| situational_riddle | 100 |

## Exact leakage (auto-exclude policy)

**Count: 0**

No normalized exact matches to graded stems.

## Soft flags (Jaccard ≥ 0.55 or containment) — review only, not auto-exclude

**Count: 0**

None.

## Scenario / fragment flags — review only

**Count: 0**

None.

## Verdict

**OK to approve** under exact-only policy; no soft/fragment flags either. Next: flip `review_status` to `approved`, build hybrid train/dev, prepare-qa.
