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

### Train and reuse the BPE tokenizer

A tokenizer trained on every `Character` and `Line` value in the full Star Wars
corpus is included at `models/star_wars_4096.bpe.json`. It contains 4,096 vocabulary
entries and 3,840 ordered merges. Use it immediately without the training dataset:

```bash
uv run python src/bpe_tokenizer.py --load models/star_wars_4096.bpe.json --text "Hello, café🙂!"
```

To reproduce the model, place `SW_EpisodeIV_VI.json` at the project root and run:

```bash
uv run python src/bpe_tokenizer.py --vocab-size 4096 --save models/star_wars_4096.bpe.json
```

The vocabulary size is configurable and defaults to 4,096, including the 256
initial byte tokens. Model files contain hexadecimal token bytes and ordered
merge rules in JSON. Other generated `models/*.bpe.json` files are ignored by Git.

In Python, use `tokenizer.save(path)` and `BPETokenizer.load(path)`.

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
