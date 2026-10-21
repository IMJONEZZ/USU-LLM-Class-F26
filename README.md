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

For the local vector database and RAG notebook:

```bash
uv sync
uv run marimo edit marimo_notebooks/vectordb.py
```

The project selects CUDA 12.8 PyTorch wheels on Windows and Linux. The notebook
requires an NVIDIA GPU and checks CUDA before creating embeddings. Chroma stores
its data locally under `data/chroma/` and saves evaluation results to
`data/vectordb_results.json`. Model weights download on
first use; the Llama Instruct model requires Hugging Face access and authentication
with `uv run hf auth login`. Edit the notebook's `qa` dictionary to change the
questions and draft answers. The cells build the database, demonstrate retrieval,
and compare Llama Instruct answers with and without retrieved context automatically.

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
