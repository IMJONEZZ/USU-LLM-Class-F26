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

### Prompt optimization experiment

On native Windows, `uv sync` installs CUDA 12.8 PyTorch from the explicit PyTorch
index. The notebook requires a working NVIDIA driver and checks actual model
placement on CUDA. DSPy 3.3 uses a small custom `BaseLM` backend to call the single
Transformers model directly; no model server or API key is needed. A small
single-output DSPy adapter accepts plain text because this Qwen model does not
reliably emit DSPy's field markers or JSON. The same adapter renders the original
and optimized instructions and demonstrations.

```powershell
uv sync
uv run marimo edit marimo_notebooks/prompt_optimization.py
uv run marimo export script marimo_notebooks/prompt_optimization.py -o src/prompt_optimization.py
uv run python src/prompt_optimization.py
```

`src/prompt_optimization.py` is the tracked, tested implementation. The Marimo
notebooks are ignored. After exporting notebook changes, keep execution inside
`main()` and the `if __name__ == "__main__"` guard: raw Marimo exports execute
their cells on import, which would start model loading during test collection.
Unit tests import the source helpers; the CUDA experiment in `main()` is verified
separately with real models.

The code cells run a training-only smoke test and the original-prompt baseline.
Set `run_optimization = True` in the configuration cell to enable the small
MIPROv2 search and final comparison. Keep this false until the smoke test and
baseline work. Expected-answer embeddings are cached separately for optimization
and testing. Both programs use identical DSPy formatting and greedy decoding.
Saved prompts and settings go into `artifacts/prompt_optimization/<run>/`.
Result CSVs go into `results/prompt_optimization/`, replacing the previous results
when rerun. Cosine scores measure semantic similarity,
not exact correctness; the 10-example test comparison is exploratory.
After optimization, `metrics_by_split.csv` reports baseline and optimized means
and their differences for train, validation, and test. Per-riddle answers and
scores are saved in each split's comparison CSV and `all_splits_comparison.csv`.
Training and validation scores describe data used during optimization; only the
test split is held out.
