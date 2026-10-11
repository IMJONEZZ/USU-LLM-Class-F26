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
