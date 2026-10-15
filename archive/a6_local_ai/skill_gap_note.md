# Skill gaps after public-only training

Frozen baseline: [`adapters/public_only/`](adapters/public_only/) (keyword 2/10; priority Q1/2/3/5/9 still 0/5).

Public train mix (~7182): StrategyQA 2053, FalseQA 3394, TruthfulQA 717, BrainTeaser 1018. That mix teaches some lateral/misconception habits but under-covers the skills below. **Do not** put graded stems (or near-duplicates) into training — `GRADED_QUESTIONS` leak gate in `src/trainer.py` stays absolute.

## Stem → skill → next data action

| Stem | Observed failure (public adapter) | Needed skill | Why public mix under-serves it | Next data action |
| ---: | --- | --- | --- | --- |
| Q1 | Scaled people into wrong time ("10 hours") | False proportional: more actors does not shrink event duration | StrategyQA is multi-hop facts; rarely "refuse fake scaling" | Synth: concerts/plays/movies where audience size is irrelevant to length |
| Q2 | Partial sense ("weight not currency") but misses full compare | Homonym / currency vs mass | FalseQA is premise-false, not word-sense traps | Synth: pound/yen/"grand"/oz ambiguities with short disambiguation answers |
| Q3 | Literal nonsense ("boy is a tree") | Situational riddle / holiday cue | BrainTeaser helps style; Christmas-tree cue is sparse | Synth: everyday scenes that resolve to a common holiday/event (not the graded stem) |
| Q4 | Invented medical harm | Debunk misconception without scare | TruthfulQA-ish; model still drifted to harm scripts | Synth: "mostly harmless habit" misconceptions with calm short answers |
| Q5 | Rejected premise ("no sharks") instead of answering safety | Absurd premise → still answer the asked bit | FalseQA often denies premise; graded ask needs the upstairs safety answer | Synth: absurd basement/pool/zoo premises where the asked location is still safe |
| Q6 | Tongue-twister script | Hard resource constraint on a joke | Jokes dominate; constraint answers are rare | Synth: jokes with explicit global resource caps ("at most N") |
| Q7 | Stale "Joe Biden" | Post-cutoff current fact | Public corpora cannot teach "current president" reliably | Synth or curated post-cutoff facts (small, carefully dated); accept residual risk |
| Q8 | Pass (keyword) | Myth yes/no with bronze automaton framing | Covered enough by public mix | Keep public examples; avoid regressing in hybrid |
| Q9 | Said "two Ls" | Character / letter counting | Almost absent from public sources | Synth: count letters/digits in ordinary words (not "parallel") |
| Q10 | Pass (keyword; answer still messy) | Classic riddle recall | Partially covered | Prefer short correct sphinx-style paraphrases in any synth; don't overfit stem |

## Priority for hybrid data volume

Focus ChatGPT generation on skills for **Q1, Q2, Q3, Q5, Q9** first (~50–150 short pairs each), then Q4/Q6, then a thin Q7 slice if desired.

## Lever (not more of the same)

More epochs on the same public JSONL are unlikely to teach letter-counting or currency ambiguity. **Skill coverage in the training mix** is the main lever; keep [`adapters/public_only/`](adapters/public_only/) frozen for A/B against the next hybrid run.
