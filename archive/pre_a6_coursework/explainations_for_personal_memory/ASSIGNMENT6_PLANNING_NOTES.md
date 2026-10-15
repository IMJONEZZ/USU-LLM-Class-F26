# Assignment 6: Planning notes (before code)

These notes complete the pre-coding planning steps from the Assignment 6 plan. They are **personal preparation**, not a finished PR Action Plan comment. Per course rules, rewrite Decision 1 / Decision 2 in **your own words** when you post the first PR comment; do not paste this file unchanged.

**Deliverables reminder (20 pts):** PR + Action Plan essay (10) · finetuned Llama-3.2-1B + `answers.ipynb` + 1-min recording (5, max 5 of 10 questions) · tests + ruff/CI (3) · explain/reflect (2).

**Hard constraint:** You may not train directly on the graded questions.

---

## 1. Skill table for the 10 graded questions

| # | Question (short) | Skill label | What to teach *indirectly* | Priority for first 5 pts |
| --- | --- | --- | --- | --- |
| 1 | Opera: 60 people / 1 hour → 600 people? | Lateral thinking / false proportion | Parallel performance ≠ linear work; “time stays the same” pattern on similar setups | **Primary** |
| 2 | Pound of feathers vs British pound | Units / wordplay | Currency “pound” vs weight “pound”; unit-disambiguation Q&A | **Primary** |
| 3 | Tree + boxes in living room | Lateral thinking / holiday scene | Christmas morning / decorated tree scenarios (not the graded stem) | **Primary** |
| 4 | Crack knuckles a lot? | Myth-busting / health lore | Common medical myths (arthritis claim vs reality) | Secondary |
| 5 | Shark in basement pool → upstairs safe? | Lateral thinking / spatial safety | Separated spaces; “threat is downstairs, upstairs OK” style puzzles | **Primary** |
| 6 | Woodchuck with only 5 lb of wood | Cap / constraint reasoning | Answers bounded by available material, not the tongue-twister max | Secondary |
| 7 | Current US President | World knowledge / time-sensitive fact | Recent civic facts; short factual Q&A (stale risk for 1B + old cutoffs) | Stretch |
| 8 | Was Talos alive? | Myth/fiction fact | Greek myth / Jason & Argonauts bronze guardian; yes/no with brief reason | Secondary |
| 9 | How many Ls in “parallel”? | Character counting / careful reading | Letter-count drills on similar words (avoid training on “parallel” itself) | **Primary** |
| 10 | Riddle of the Sphinx + all answers | Classic riddle + exhaustive answers | Sphinx riddle variants + man/morning-noon-night; “all satisfying answers” needs breadth | Stretch |

### Priority rationale (target ~5 pts first)

Focus first on skills that a 1B model can learn from **patterned instruction data** without needing a fresh news crawl or long multi-answer essays:

1. Opera false-proportion (1)
2. Pound wordplay (2)
3. Christmas living-room scene (3)
4. Shark / upstairs safety (5)
5. Letter counting (9)

**Secondary / stretch:** knuckles myth (4), woodchuck cap (6), Talos (8), president (7), Sphinx exhaustive set (10). Attempt these after the primary five look solid on **proxy** items.

### What “correct” means (working bar for `answers.ipynb`)

| Item | Accept gist if… |
| --- | --- |
| 1 | Same duration (1 hour), not 10× longer |
| 2 | Distinguishes weight vs currency; feathers not heavier as mass than a £1 coin’s “pound” confusion is resolved correctly |
| 3 | Christmas / holiday tree with presents |
| 4 | Does not cause arthritis (or equivalent myth-bust) |
| 5 | Yes, upstairs is safe (shark confined downstairs) |
| 6 | At most 5 pounds (or “only 5 pounds”) |
| 7 | Correct name as of grading time (verify before submit) |
| 8 | Yes (or myth-appropriate “animated construct / living bronze”) with clear affirmative |
| 9 | Three (parallel → p-a-r-a-**ll**-e-**l**) |
| 10 | States the riddle and man / stages-of-life answer; extra valid variants if asked |

---

## 2. No-direct-train data policy

### Policy (commit this before generating data)

1. **Banned from train/dev:** Exact text of any of the 10 graded questions; trivial whitespace/punctuation-only variants; answer lines paired with those exact stems.
2. **Banned in spirit:** Near-paraphrases that keep the same distinctive entities and structure (e.g. “60 people play an opera in 1 hour; how long for 600?”). If a grader would recognize it as the homework item, exclude it.
3. **Allowed:** Different stems that teach the *skill* (other false-proportion puzzles; other currency-vs-weight traps; other holiday lateral puzzles; other letter-count words; short myth blurbs about Talos-like figures; general medical-myth Q&A).
4. **Leak check (required before every train run):** Script or notebook cell that asserts none of the 10 graded strings (normalized: lowercased, collapsed whitespace) appear as substrings in any training field. Keep the list of graded stems in a single `GRADED_QUESTIONS` constant used only for leak checks and final eval—not for training labels.
5. **Proxy eval set:** 10–20 *similar-but-not-identical* items (2–4 per primary skill). Tune hyperparameters on proxies only. Touch the graded 10 only for `answers.ipynb` after a smoke + proxy pass.
6. **Provenance log:** Short `data/a6/README.md` (or notes here) listing sources, generation method, date, and “leak check passed.”

### Recommended data mix (draft)

| Bucket | Approx. share | Example sources / method | Format |
| --- | --- | --- | --- |
| Lateral / situation puzzles | ~40% | Hand-written or synthetic cousins of opera / shark / Christmas skills; public riddle collections **after** filtering graded stems | Instruction → short answer |
| Wordplay / units / counting | ~25% | Currency-vs-mass, letter counts on words other than “parallel” | Same |
| Myth-bust + factoid Q&A | ~20% | Short explanations (knuckles, myths, Talos-related myth text you write from memory/public domain summaries) | Same |
| General short instruction following | ~15% | Tiny Alpaca-style or self-written commands so the model keeps answering concisely | Same |

**Size target for smoke:** 50–150 examples. **Size target for real run:** a few hundred to ~1–2k short pairs—enough for LoRA on 1B without overnight corpus hunting. Prefer quality and skill coverage over dumping huge unrelated chat logs.

**Optional tooling:** [distilabel](https://github.com/argilla-io/distilabel) or a small local script to expand templates you wrote by hand; always run the leak check after generation.

### Example template shapes (not graded stems)

```text
### Instruction:
A choir of 40 singers finishes a hymn in 3 minutes. How long for 400 singers to finish the same hymn if they sing together?

### Response:
3 minutes. They perform in parallel, so more singers do not make the piece take longer.
```

```text:
### Instruction:
How many times does the letter "e" appear in the word "excellence"?

### Response:
4
```

(Store real files under something like `data/a6/train.jsonl` + `data/a6/proxy_eval.jsonl`.)

---

## 3. Stack ADR drafts (rewrite for PR)

Use these as **options you considered**. Pick the recommendation only if you agree; then phrase Decision 1 / Decision 2 yourself in the PR.

### Decision area A — Where / how you train

| Option | Pros | Cons |
| --- | --- | --- |
| **A1. Reuse A5 Docker + Unsloth LoRA on local GPU** (recommended default) | Already working on this machine; pinned deps; same `meta-llama/Llama-3.2-1B`; fast iteration | Image/Makefile still named a5; must rewire task away from span recovery |
| A2. Unsloth Colab notebooks (conversational / Alpaca) | Official course link; free GPU | Weaker reproducibility in repo; harder CI story; download/upload adapters |
| A3. Plain HF `Trainer` / TRL SFT on host venv | Simpler mental model | May fight CUDA/Unsloth pins you already solved in Docker |
| A4. SageMaker | Course-listed; scalable | Setup overhead for a 1B LoRA homework |

**Draft choice:** A1 — LoRA (not full finetune) on Llama-3.2-1B via existing training image patterns. Full finetune is unnecessary VRAM/time for this skill mix.

**Base weights:** Start from the same base `meta-llama/Llama-3.2-1B` you used in A5 unless an instruct checkpoint is clearly easier for short Q&A *and* you accept license/access differences. Document which ID you load.

### Decision area B — Data format and objective

| Option | Pros | Cons |
| --- | --- | --- |
| **B1. Alpaca-style Instruction/Response** (recommended) | Matches Unsloth Alpaca notebooks; easy jsonl; good for short answers | Less “chatty” than conversational template |
| B2. Chat / conversational template | Natural for multi-turn | Overkill for single-shot graded Qs; template bugs |
| B3. Plain completion (question\\nanswer) | Minimal | Weaker instruction following |

**Loss:** Prefer masking to train on the **response tokens** (answer-only / completion-only), same idea as A5 answer-only loss—so the model learns answers, not to regurgitate the question.

**Stages:** One mixed skill dataset first (simpler). Only add a second stage if proxies show one skill collapsing.

### Decision area C — Mapping to PR template slots

Suggested pairing when you write the PR (in your words):

- **Decision 1:** Training stack + LoRA vs full (area A).
- **Decision 2:** Data strategy / no-leak policy + prompt format (area B + §2).

---

## 4. Repo / CI decisions

### Branch base

| Option | Pros | Cons |
| --- | --- | --- |
| **C1. From latest `main`, new branch `talha_shahzad_6`** | Clean homework PR; matches PDF “latest main” | Must reintroduce trainer/Docker pieces you need |
| **C2. From `talha_shahzad_5`, adapt trainer** (recommended default) | Keep Unsloth/Docker/LoRA/test patterns; faster path to green GPU runs | PR may include A5 history; replace span-recovery objective carefully so A6 story is clear |

**Draft choice:** C2 for speed, with a clear commit message that A6 changes the training *task* to trick/riddle Q&A. If the course wants a thin PR vs `main` only, switch to C1 and copy the training infra forward deliberately.

### Keeping CI green (CPU)

CI runs: `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest` with `--cov=src --cov-fail-under=80`.

**Policy:**

1. GPU / Unsloth imports only inside train/infer code paths (same pattern as current A5 `trainer.py`).
2. `tests/test_trainer.py` uses mocks/fakes: dataset prep, leak filter, prompt formatting, loss-mask shapes, CLI `--help` / prepare dry-run—**no** real GPU train in CI.
3. Optional `@pytest.mark.gpu` file for local/Docker only; not required for CI.
4. Replace A5-specific span tests that would fail under the new task; keep coverage ≥ 80% on `src`.

### Minimal `Trainer` surface for tests

Enough to show “works as intended” without a GPU:

- Load/validate jsonl → example records
- Apply chat/Alpaca formatting
- Enforce leak check against graded list
- Build labels / attention masks (answer-only) on tiny fake token ids
- Checkpoint selection helper (e.g. best proxy loss) if you keep that logic
- CLI entrypoints: `prepare`, `train` (train may early-exit without GPU with a clear error)

---

## 5. Smoke train → notebook → presentation

### Smoke train (do this before long runs)

1. Build/run training environment (reuse `Dockerfile.training` / Makefile patterns; rename targets to `a6-*` only if you have time—function first).
2. Train 50–150 examples, 1 small epoch or few hundred steps, LoRA ranks modest (e.g. start near A5 defaults unless OOM).
3. Eval on **2 proxy items** (one lateral, one letter-count). Success = sensible short answers, not perfection.
4. If collapse/gibberish: fix formatting or LR before scaling data.

### `answers.ipynb` outline

1. Install/load path notes (adapter dir, `HF_TOKEN` if needed).
2. Load base + LoRA adapter; set `eval()` / greedy decode.
3. Cell list: each graded question → `generate` → print raw answer.
4. Optional: one markdown cell stating leak-check was run and graded stems were not in train.
5. Freeze outputs (run all cells before push) so graders see printed answers without a GPU.

**Inference defaults (draft):** greedy or low-temperature; max new tokens ~64–128 (higher for Sphinx); no heavy system prompt unless proxies need it.

### Iteration rule

| Symptom | Try first |
| --- | --- |
| Wrong skill, confident | Add proxy-similar train pairs for that skill |
| Right idea, verbose/wrong format | More short-answer examples; tighten response style |
| Gibberish / forgot English | Lower LR / fewer steps; add general instruction pairs |
| Almost correct on graded | Prefer data/prompt fixes over memorizing the graded stem |

### 1-minute presentation outline

| Seconds | Content |
| --- | --- |
| 0–15 | Goal: finetune 1B to answer trick questions **without** training on the homework list |
| 15–35 | Data policy + skill mix (show one proxy example, not a graded stem) |
| 35–50 | Stack: Unsloth/LoRA + answer-focused loss; point at `answers.ipynb` |
| 50–60 | Result: how many of 10 looked correct; one live or recorded sample answer |

Record that talk and attach/link it in the PR (per PDF).

### Q4 reflection prompts (write after results)

- What differed from this plan?
- Why these tests (leak check, format, mask, CLI)?
- If more time: more proxy coverage, instruct base, better Sphinx/president handling, `a6-*` Makefile cleanup.

---

## 6. Suggested work order (execution checklist)

- [ ] Branch (`talha_shahzad_6` from chosen base); open PR; **first comment** = Action Plan essay in your words; leave it unedited.
- [ ] Confirm CI green on the empty/early PR.
- [ ] Write `data/a6` train + proxy jsonl; run leak check.
- [ ] Adapt `src/trainer.py` for Q&A LoRA (GPU-gated).
- [ ] Smoke train → proxy check → full train.
- [ ] Fill `answers.ipynb` with printed graded answers.
- [ ] `tests/test_trainer.py` + ruff; CI green.
- [ ] Inline code notes; Q4 PR comment; 1-min recording.

```text
skills → pick 5 → data policy → stack → format → smoke → notebook → tests → reflect
```

---

## 7. Resources (from assignment + local)

- [HF Transformers training](https://huggingface.co/docs/transformers/training)
- [HF SageMaker](https://huggingface.co/docs/sagemaker/train)
- Unsloth Colab: Llama 3.2 conversational; Alpaca-style notebooks (links in Assignment 6 PDF)
- [distilabel](https://github.com/argilla-io/distilabel)
- ADR templates linked in the PDF
- Local infra reference: `src/trainer.py`, `Dockerfile.training`, `Makefile` `a5-*` targets (patterns only; new task)

---

## Ownership reminder

Skill labels and the recommended defaults above are a **starting draft** for your planning. Before the graded Action Plan comment, replace recommendations with decisions you actually believe, including pros/cons you can defend in class.
