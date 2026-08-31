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
