# USU LLM Class — Fall 2026

Homework-submission repository for USU LLM Class DSAI-5810/6810.

> Assignments and course materials will be added throughout the Fall 2026 semester.

## Contributing

### Set up your environment

Install `uv` using the instructions found here: <https://docs.astral.sh/uv/getting-started/installation/>

Then sync the project:

```bash
uv sync
```

### Start experimenting

Run the entrypoint script:

```bash
uv run src/main.py
```

If you prefer a Jupyter notebook:

```bash
uv run --with jupyter jupyter lab
```

### Set up pre-commit (optional)

CI runs the project checks for every push and pull request. To run those checks before committing, install the hooks:

```bash
uvx prek install
```

### Run checks locally

```bash
uv run ruff check
uv run ruff format --check .
uv run pytest
```

### AI assistance

The course supports Claude Code and Codex. Both assistants follow the same student-learning and academic-integrity expectations in [AGENTS.md](AGENTS.md).

## Decisions

- M0: config tests live in `tests/test_data.py` (layout has no `test_config.py`).
- M0: `LEARNING_RATE = 2e-4`, the value the old repo actually trained with (its plan said 1e-5; the code and log used 2e-4).
- M0: `EPOCHS = 3`, `TEACHER_MODEL`, `JUDGE_MODEL` and `CATEGORIES` are placeholders, confirmed before M2.
- M0: round 0 is valid for `adapter_path`; negatives raise `ValueError`, non-ints (including `bool`) raise `TypeError`.
- M0: API key env var is `GOOGLE_API_KEY`.
- M0: deleted `src/main.py` and `tests/test_main.py` (not in the layout).
- M2: dropped `character_counting` from `CATEGORIES` (no public source has such questions and training data is not augmented); test-set rows must not use it.
- M2: BrainTeaser is converted once, by hand, to JSONL (the source is an encrypted zip of pickles, and unpickling untrusted data runs code). The JSONL lives in `assets/brainteaser.jsonl`, a file outside the 20-file layout, so a fresh clone can rebuild without the pickle.
- M2: `stratified_sample(rows, n, seed)` takes `n` per category; categories smaller than `n` return all their rows.
- M2: row schema is `id, question, answer, accepted_alternatives, category, source` plus optional `chain`.
- M2: Misguided Attention removed entirely (no answer key); `misguided_answers.jsonl` leaves the layout and `lateral_thinking` comes from BrainTeaser sentence puzzles only.
- M2: loaders live in `src/sources.py` (user-approved; outside the 20-file layout because `data.py` would pass 300 lines). Dependency order: config -> data -> sources.
- M2: loaders keep the original long-form answers; the teacher rewrites keys to short form in M3 (original kept as an alternative) and the user spot-checks a sample.
- M2: category mapping: misconception = TruthfulQA Misconceptions, Misconceptions: Topical, Superstitions, Myths and Fairytales, Mandela Effect; time_sensitive = TruthfulQA Indexical Error: Time + FreshQA fast/slow-changing non-false-premise; false_premise = FalseQA label 1 + FreshQA false_premise TRUE; pun_ambiguity = BrainTeaser word puzzles; lateral_thinking = BrainTeaser sentence puzzles. Everything else is dropped.
- M2: FreshQA rows other than never-changing are kept only when `effective_year == REFERENCE_YEAR`.
- M2: BrainTeaser `_SR`/`_CR` variants are dropped; rows whose answer is not `choice_list[label]`, with duplicate choices, or whose answer is "None of above." are dropped with a warning.
- M2: duplicate questions (normalized) are dropped across sources, first source wins (order: TruthfulQA, FalseQA, BrainTeaser, FreshQA).
- M2: blank questions or answers raise instead of being skipped.
- M2: TruthfulQA rows whose best answer is an abstention ("I have no comment") are dropped, and abstentions are stripped from `accepted_alternatives`, so abstaining can never be scored correct.
- M2: FreshQA time_sensitive rows keep all fast- and slow-changing facts for `effective_year == REFERENCE_YEAR` (77 fast + 9 slow); answers are right as of the April 2026 snapshot only and can be stale. Revisit if the time_sensitive test answers use a different convention.
- M2: `assets/` and `data/` are both gitignored on purpose: no dataset is committed, including the hand-written test set and `brainteaser.jsonl` (overrides the spec line "assets/ is not ignored"). A fresh clone must recreate `assets/` by hand; `git check-ignore` tests pin this.
- M3: teacher = `gemini-3.1-pro-preview` (a preview model), judge/key-shortener = `gemini-3.8-flash`; chosen from Google's models page, not verified with an API call.
- M3: chain length bounds (120 to 1500 characters) are provisional until checked against real teacher output.
- M3: `scripts/pipeline.py` adds the repo root to `sys.path` itself because the project is not an installed package.
- M3: every API attempt is timed (seconds + outcome) in an in-memory call log; the run summary has per-model timing and per-row seconds, and the progress bar shows live API call count and slowest call.
- M3: TEACHER_MODEL changed to `gemini-3.8-flash` because `gemini-3.1-pro-preview` returned 429 RESOURCE_EXHAUSTED (no quota on this key); one real flash chain took 20.7 s and passed the filter.
- M3 (simplified at the user's request): one teacher call per row. The prompt gives the question and the correct answer and asks for the five-part chain; there is no blind solving, no fallback, no key shortening and no `answer_given` tag (this replaces the spec's blind-solve design).
- M3: a chain is kept only if it has the five sections in order and a non-empty final answer; for keys of at most 60 characters the final answer must also match the key or an alternative; for longer keys the teacher writes a short final answer itself.
- M3: up to 3 attempts per row; rate-limit, timeout and server errors are retried twice (2 s apart) first; a daily-quota 429 is its own kind (`quota`) and is not retried.
- M3: only content-level failures (empty, blocked, truncated) are skipped; auth, quota, bad-request, rate-limit, timeout and server errors that survive the retries stop the run with the real message and every finished chain is already in `data/chains.jsonl`, so a rerun continues.
- M3: `TEACHER_SAMPLES`, `SHORT_KEY_MAX_WORDS` and the fallback constant are gone; `TEACHER_ATTEMPTS = 3` replaces them. `TEACHER_MODEL = gemini-3.5-flash-lite`: `gemini-3.1-pro-preview` has no free quota on this key, `gemini-3.8-flash` allows 20 requests per day and takes about 20 s per call, flash-lite answered in about 1 s and its sample chains read well (3 real rows checked).
- M4: the hand-written test set's own categories were relabelled to the 5 training categories (false_assumption -> false_premise, wordplay -> pun_ambiguity, common_misconception -> misconception, knowledge_currency -> time_sensitive, definitional_trap and lateral_thinking -> lateral_thinking); the counting question (`ls-in-parallel`) was removed; `source` defaults to `hand_written`; the time_sensitive row carries `reference_year: 2026`.
- M4: `test-data` checks the test set (schema, five-part reference chain whose final answer equals the key or an alternative, unique ids, reference year) and records its SHA-256 in `data/test_set_hash.jsonl`; `validate` also checks leakage into train and eval and warns when the test file changed since the hash was recorded. Sphinx exclusion is by id via `EXCLUDED_TEST_IDS` (empty: the sphinx row is not in the file).
- M4: near-duplicate threshold is 0.9 (difflib ratio on normalized questions); on the real data the closest test/public pair other than the exact leak was 0.68.
- M4: a real leak was found: test question `cracking-knuckles-effect` is identical to TruthfulQA `tqa-0744`. `train-data` now drops public rows that duplicate (or nearly duplicate) a test question before sampling.
- M5: the judge (`JUDGE_MODEL`) answers JSON with `natural` and `answer_unchanged`; a variant is kept only if both are true; invalid output, a failed judge call (empty, blocked, truncated, timeout, server) or a rejection drops that variant into `data/eval_review.jsonl` with its reason; auth, quota, rate-limit and bad-request errors stop the run; nothing already generated or judged is lost because it is in `data/eval_candidates.jsonl`.
- M5: deviation from the spec: the judge does not write revised chains; every variant reuses its source question's reference chain unchanged.
- M5: `data.py` is now over 450 lines (the 300-line guideline) because the spec assigns these functions to it; kept as is, noted here as the open question.
- M5: `JUDGE_MODEL = gemini-3.5-flash-lite` (same reason as the teacher: `gemini-3.8-flash` allows 20 requests a day and takes about 20 s per call).
- M5: the judge sees a pool of `per_question * 3` candidates per question and stops once `--per-question` variants are kept; the first real run capped the pool at the kept target and left 6 of 8 questions without variants (the judge rejected 5 of 7).
- M5: a 429 carries the wait the server asks for (RetryInfo); the client waits that long (at least 2 s) up to 90 s and gives up at once on longer waits (daily quota); the first real eval run stopped because its judge calls hit the per-minute limit.
- M5 (changed after a real run): the teacher model writes the rewordings (one call per question, JSON `rewordings`) instead of `nlpaug` WordNet synonyms. On the real test questions only about 14% of WordNet candidates survived the judge (6 kept of 42, 4 of 8 questions and 2 categories left empty) because synonyms ignore word sense; this replaces `nlpaug` and `nltk`, which were removed from the project.
- M5: code still drops a rewording if it is unchanged after normalizing, a duplicate, blank, contains a template marker, or changes a number, a unit or the negation count; the judge then keeps it only if it is natural and the answer is unchanged.
- Never pay twice (user rule): every generated chain is saved the moment it exists in the permanent store `data/chains.jsonl`, and a row whose generation failed is saved too and never retried; `data/train.jsonl` is derived from the store, so rewriting it costs nothing and there is no `--force`. A stored row whose question or answer later changes upstream stops the run before any call (delete its store line to regenerate it on purpose). The sampler is prefix-stable (a larger `--per-category` always contains the smaller sample), so extending a run only generates the new rows.
- Never pay twice, eval side: `data/eval_candidates.jsonl` stores each question's rewordings and every judge verdict; a rerun uses the cache first and only paraphrases or judges what is missing (a failed judge call is not cached because nothing was received); a lower or equal target spends nothing; `data/eval.jsonl` and the review file are derived from the store.
- M6: GPU = L4 on Modal for training and evaluation (user choice, changed from the cheapest, T4): 24 GB, $0.80/hour, supports bf16 and is about twice as fast as a T4; the cheapest that fits is still T4 ($0.59/hour) and the audit prints both. Keeping eval and training on the same GPU type keeps the baseline comparable.
- M6: the token audit (`validate --tokens`) counts the start token and the end-of-text token with the real Llama 3.2 tokenizer (`tokenizers`, loaded from the Hub or a local file), reports prompt, response and total separately for train, eval and test, fails if any example is longer than `MAX_SEQ_LENGTH`, and writes `data/token_audit.json`; memory is a rough estimate (weights, LoRA and optimizer, activations, logits, 1.5 GiB overhead) that the two-step probe in M8 must confirm.
- M6: `MAX_SEQ_LENGTH` is the next multiple of 64 at or above the longest example across train, eval and test.
- M7 (changed at the user's request): the baseline prompt is zero-shot: a written step-by-step instruction that describes the five parts (the same `CHAIN_INSTRUCTIONS` text the teacher uses), then the question, then the response marker; no examples from the training data (the spec said few-shot).
- M7: generation is greedy, left-padded, batch size 8, at most 384 new tokens (the longest real response so far is 213 tokens), cut at the first `Final answer:` line; evaluation uses its own context size (`EVAL_MAX_SEQ_LENGTH = 2048`) separate from the training limit; a response cut off before its final answer is wrong and a format failure.
- M7: scoring: accuracy overall and per category (a category with no rows is "undefined", never 0), format compliance (five sections in order and a non-empty final answer), and average response length in characters; `find_weak_categories` flags a category whose accuracy is at least `gap` below the mean of the categories that have an accuracy (the gap is an argument with no default, to be chosen from real results in M9; the earlier minimum of 3 distinct source questions per category was removed because the eval set has only 8 source questions, so four categories could never qualify).
- M7: the baseline loads the model 4-bit with Unsloth exactly as training will (code taken from the earlier project), on the same L4 GPU; result files hold model, split, sizes, batch size, versions, metrics, timing, GPU summary and settings; per-row outputs go to `data/<name>_outputs.jsonl`; the `test` split is refused a second time without `--force`.
- M7: Modal commands go through the Makefile (`make modal_run ARGS="..."`) which sets `PYTHONPATH=$HOME/.pythoncustomize`: that folder's `sitecustomize.py` runs `truststore.inject_into_ssl()`, the fix for the proxy certificate problem that also blocks Modal; the Modal image reuses the earlier project's exact package set (Python 3.11, torch 2.12.1, unsloth 2026.9.14, transformers 5.5.0 and so on).
- M7: the container runs Python 3.11 but the project targets 3.14, whose formatter turns `except (A, B):` into `except A, B:` (3.11 rejects it); the files the container imports avoid that form and a test guards it.
- M7: `scripts/modal_app.py` has one GPU function (`evaluate_on_gpu`) and one entrypoint (`eval_split`, run as `::eval_split`); the spec's `train` function comes in M8.
- Never pay twice, no-generate mode: `train-data --no-generate` writes `data/train.jsonl` from the chains already in `data/chains.jsonl` for the requested rows and makes no API calls (rows without a chain yet are reported as pending); it refuses to write an empty file. Added when the free daily quota (500 requests per model) ran out at 389 of 450 chains.
- M6: `MAX_SEQ_LENGTH = 320`, from the token audit of 383 real training chains (longest 296 tokens: median 138, p90 198; prompt max 128, response max 253); re-run `validate --tokens` after the last chains are generated and raise it if the audit says so.
- M8: training choices (user): early stopping on with patience 1 (stops once eval loss worsens after an epoch; the best-eval-loss checkpoint is still the one kept), 4 examples per pass x 4 accumulation = 16 per update, 3 epochs, learning rate 2e-4 (the earlier project's value), seed 0; with 383 rows that is about 24 updates per epoch.
- Shared instruction text fixed (user): `CHAIN_INSTRUCTIONS` read "...Verify:, Final answer: The Check section..." with no sentence break; it now reads "...Final answer:. The Check section...". The baseline prompt and any new teacher call use the new text; teacher chains already in `data/chains.jsonl` were written with the old wording and were not regenerated; the fine-tuned prompt is the plain question and is unaffected. The baseline was rerun after this change, and the earlier baseline is kept under `results/old_prompt/`.
- M8: learning rate lowered 2e-4 -> 1e-4 (user) after round 0 at 2e-4 early-stopped at epoch 2: train loss fell 1.909 -> 1.420 while eval loss rose 2.340 -> 2.396, the best checkpoint being epoch 1. Changing it changes the hyperparameter hash, so round 0 has to be rerun at 1e-4 before any later round.
- M8/M9 results that led to the change below (judged by Gemini): round 0 at lr 2e-4 scored 6/21 (28.6%); round 1 (weak categories false_premise, misconception, time_sensitive repeated twice plus 0.4 replay, 720 rows) scored 3/21 (14.3%) with lateral_thinking falling from 4/9 to 0/9, and its best eval loss (2.440) was worse than round 0's (2.340). Eval loss bottomed at the end of epoch 1 (about step 29) in both rounds.
- M8: change after those results (user): learning rate 1e-4, and evaluate, save and log every `EVAL_EVERY_STEPS` = 10 optimizer steps instead of once per epoch, early stopping patience 3 (counted in evaluations, about one epoch). `save_steps` must stay a multiple of `eval_steps` for the best checkpoint to be restored. Records carry `step` and a two-decimal `epoch` (the old `round(state.epoch)` would have given several evaluations the same epoch). The probe now trains into `/checkpoints/probe` instead of the real round folder. These settings change the hyperparameter hash, so the rounds recorded at 2e-4 are no longer valid; round 0 is retrained and the volume folders `round_0`, `round_1` and `probe` are cleared first, because stale `checkpoint-N` folders would be mixed into the trainer's checkpoint cleanup.
- M8: checkpoint selection changed to format first (user). The new round 0 (lr 1e-4, eval every 10 steps) had its lowest eval loss at step 20 (2.284) with format compliance 5%, so the restored adapter produced rambling answers (format 4.8%, 1/21); steps 30 to 50 had formats 95%, 100%, 100% and eval losses 2.313, 2.340, 2.399. The kept checkpoint is now the lowest-eval-loss one among evaluations with format compliance exactly 100% (on those numbers, step 40). It is implemented as one number, `eval_format_gated_loss` = eval loss + `FORMAT_PENALTY` (100) x (1 - format compliance), added to the evaluation metrics by the per-step callback (which must run before early stopping reads it) and used as `metric_for_best_model`; because it improves as the format improves, early stopping keeps waiting before the format is learned. If no evaluation reaches 100% the highest-format checkpoint (then lowest loss) is still saved, with a printed warning and a `format_warning` note in the round's record (user chose this over refusing or over falling back to lowest eval loss, which on the earlier run would have kept the 5%-format step 20). `checkpoint_selection` is in the hyperparameters, so the hash changes and the round recorded before this change is no longer valid.
- M8 (superseded by the lines above): the trainer evaluates and saves once per epoch and logs the training loss once per epoch; it keeps the best 2 checkpoints; the loss is computed only on the chain and the end token (`train_on_responses_only` with the two template markers).
- M8: the per-epoch scoring hooks the trainer's evaluate event (Hugging Face computes the eval loss after the epoch-end event) and records epoch, train loss, eval loss, accuracy, format compliance, average length and per-category accuracy (undefined for an empty category); it switches the model to inference mode to generate and always back to training mode, even if generation throws.
- M8: before training the Modal function checks, on the real trainer's labels, that exactly the chain and the end token are unmasked (`check_masking`), and the pure marker-boundary check runs in the tests with the real Llama tokenizer.
- M8: `--probe` runs the whole real path on 8 training rows and 4 eval rows for 2 epochs (2 optimizer steps), saves nothing and writes nothing to `rounds.json`; the local entrypoint first prints the longest example, the memory estimate and the number of steps, and warns when `train.jsonl` still has pending rows.
- M8: round numbers start at 0; `results/rounds.json` refuses a repeated round or a gap, records the hyperparameters and their hash (M9 compares it), the example counts and the epoch results; the adapter of round n lives on the Modal volume at `/checkpoints/round_n`; `load_adapter` checks the base model and the LoRA rank and refuses gaps in the recorded rounds. Only round 0 can be trained so far.
- M8 fix found by the first real probe: inside the Modal container Unsloth returns a Hugging Face tokenizer whose `encode()` gives a plain list, while the local `tokenizers` library gives an object with `.ids`; all token counting and masking code now goes through `data.encode_ids`, which accepts both, and the tests run every one of those functions with both styles.
- M8 fix: generation sets the tokenizer to left padding only while it generates and always restores the previous side (even on an error), because the per-epoch evaluation shares the tokenizer with the trainer and must not change how the next training batch is padded.
- M8: padding. We set neither `packing` nor `padding_free`; Unsloth overrides trl's `padding_free` default to None and then turns padding-free on automatically for every model except two blocklisted families (checked in its current `main` source, not the pinned version); padding-free flattens the examples of a batch into one row, so the probe now prints `padding_free`, `packing` and the real collated batch shapes, and `check_batch_masking` checks that every contiguous trained stretch of the collated labels is exactly one example's chain plus end token, which also proves the example boundaries are not merged.
- M9 (my readings of the spec, for review): round 0 is the first training run and rounds 1 to 3 are the targeted rounds (`MAX_ROUNDS = 3`, a round beyond it is refused); a round trains on the new examples plus a seeded replay sample of rows from the non-weak categories (`replay_fraction` x the number of new rows, 0 means none, above 1 is allowed, a negative value is refused, and a request larger than the available rows takes all of them once); with no extra rows the existing rows of the weak categories are repeated `ROUND_REPEAT_CAP = 2` times each, never more than the cap; the round's rows are shuffled with the seed.
- M9: your extra rows (`--extra-data`, a JSONL file with chains) must pass the same schema and chain checks as training rows, may not duplicate a test or eval question (that stops the round), and duplicates of existing training questions are dropped with a warning; an extra row outside the weak categories is kept with a warning; a missing or empty file, a row without a chain or an unknown category all stop the round before any GPU time.
- M9: `plan_round` refuses a round beyond the maximum, a round that is already recorded or skips one, a `--resume-adapter` that is not the previous round's adapter (`/checkpoints/round_{n-1}`), and changed hyperparameters (the recorded hash of round 0 is compared); a round continues from the previous adapter by loading it as the starting model, and Unsloth skips attaching new LoRA weights when it already has them (not yet confirmed on the GPU).
- M9: the round decision reads the judged scores (Gemini yes/no per final answer, `scripts/judge_outputs.py`), not the exact-match numbers printed during training, because exact match scored 0/21 on round 0 while the judge scored 6/21 (the model's answers were right but worded differently). The order is `make train`, `make eval_finetuned ROUND=n`, `make judge ROUND=n`, `make decide ROUND=n GAP=x`; `decide` lists the weak categories and says continue or stop, and with `DROP=y` also lists categories that fell by at least that much against the previous round's judged file. Neither threshold has a default because they should come from real results. The judged file of a round must be regenerated after that round is retrained. The decision used to live in `train_round` (`--gap`/`--drop`); it was removed so there is only one decision, on one metric.
