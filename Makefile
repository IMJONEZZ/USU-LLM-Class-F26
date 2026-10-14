SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.DEFAULT_GOAL := all

# /tmp on this host often hits quota during CUDA wheel extract; keep uv temp on $HOME.
export TMPDIR ?= $(HOME)/.cache/uv-tmp
export UV_CACHE_DIR ?= $(HOME)/.cache/uv

.PHONY: all help sync

all:
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest

sync:
	mkdir -p "$(TMPDIR)" "$(UV_CACHE_DIR)"
	uv sync --group dev

help:
	@printf '%s\n' \
	  'Default: make (or make all) runs ruff check, ruff format --check, then pytest' \
	  'make sync          Recreate/repair .venv (uses $$HOME cache; avoids /tmp quota)' \
	  'Assignment 6 focused branch — older A5 make targets live in archive/pre_a6_coursework/root_files/Makefile'
