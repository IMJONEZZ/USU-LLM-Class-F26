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
- M3: only content-level failures (empty, blocked, truncated) are skipped; auth, quota, bad-request, rate-limit, timeout and server errors that survive the retries stop the run with the real message and keep the partial file, so a rerun resumes.
- M3: `TEACHER_SAMPLES`, `SHORT_KEY_MAX_WORDS` and the fallback constant are gone; `TEACHER_ATTEMPTS = 3` replaces them. `TEACHER_MODEL = gemini-3.5-flash-lite`: `gemini-3.1-pro-preview` has no free quota on this key, `gemini-3.8-flash` allows 20 requests per day and takes about 20 s per call, flash-lite answered in about 1 s and its sample chains read well (3 real rows checked).
