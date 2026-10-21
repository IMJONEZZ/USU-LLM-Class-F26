SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.DEFAULT_GOAL := all

# /tmp on this host often hits quota during CUDA wheel extract; keep uv temp on $HOME.
export TMPDIR ?= $(HOME)/.cache/uv-tmp
export UV_CACHE_DIR ?= $(HOME)/.cache/uv

.PHONY: all help sync setup-unsloth serve serve-transformers infer

all:
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest

sync:
	mkdir -p "$(TMPDIR)" "$(UV_CACHE_DIR)"
	uv sync --group dev

setup-unsloth:
	mkdir -p "$(TMPDIR)" "$(UV_CACHE_DIR)"
	uv venv --no-project --python 3.12 --allow-existing .venv-unsloth
	uv pip sync --python .venv-unsloth/bin/python requirements-unsloth.lock

serve:
	.venv-unsloth/bin/python -m uvicorn src.server:app --host 127.0.0.1 --port 8000 --workers 1

serve-transformers:
	uv run uvicorn src.transformers_app.server:app --host 127.0.0.1 --port 8000 --workers 1

infer:
	uv run python -m src.inference

help:
	@printf '%s\n' \
	  'Default: make (or make all) runs ruff check, ruff format --check, then pytest' \
	  'make sync          Recreate/repair .venv (uses $$HOME cache; avoids /tmp quota)' \
	  'make setup-unsloth Install the separate Python 3.12 Unsloth environment' \
	  'make serve         Serve the final LoRA adapter with Unsloth at localhost:8000' \
	  'make serve-transformers Run the preserved Transformers/PEFT server (CPU or CUDA)' \
	  'make infer         Print an example API request and response' \
	  'Assignment 6 focused branch — older A5 make targets live in archive/pre_a6_coursework/root_files/Makefile'
