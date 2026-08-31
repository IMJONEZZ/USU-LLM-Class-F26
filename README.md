# USU LLM Class — Fall 2026

Repository for USU LLM Class DSAI-5810/6810.

> This repo is a work in progress for the Fall 2026 semester. Assignments, notebooks,
> and course materials will be added here as the semester approaches.

## Contributing

### Set up your environment

Install `uv` using the instructions found here: <https://docs.astral.sh/uv/getting-started/installation/>

Then sync the project:

```bash
uv sync
```

### Add dependencies

If you need to add dependencies to run your code, for example cowsay, you can do so with:

```bash
uv add cowsay
```

### Change the code

Update the code to complete your homework. You can run your experiment like so:

```bash
uv run main.py
```

If you prefer working in a jupyter notebook setting you can run:

```bash
uv run --with jupyter jupyter lab
```

### Run formatting/linting/tests

```bash
uv run ruff format
uv run ruff check
uv run pytest
```
