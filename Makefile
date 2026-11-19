SHELL := /bin/bash
.DEFAULT_GOAL := all

.PHONY: all sync run test lint format clean

all: lint test

sync:
	uv sync

# Start the Gradio chatbot on 127.0.0.1:7860 (Ollama must already be running).
run:
	uv run python -m src.frontend

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

clean:
	rm -rf -- .pytest_cache .ruff_cache htmlcov .coverage .coverage.* __pycache__
	find src tests -type d -name __pycache__ -prune -exec rm -rf -- {} +
