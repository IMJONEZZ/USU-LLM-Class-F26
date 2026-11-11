SHELL := /bin/bash
.DEFAULT_GOAL := all
DATASET ?= fixtures/a10.json

.PHONY: all sync clean smoke optimize evaluate report reference

all:
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest

sync:
	uv sync --group dev

# Reset generated state while keeping dependencies and downloaded models installed.
clean:
	rm -rf -- artifacts .pytest_cache .ruff_cache htmlcov .coverage .coverage.* __pycache__
	find src tests -type d -name __pycache__ -prune -exec rm -rf -- {} +

smoke optimize evaluate report:
	uv run python -m src.prompt_optimization $@ --dataset "$(DATASET)"

reference:
	uv run python -m src.a9_reference
