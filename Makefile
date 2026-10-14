SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.DEFAULT_GOAL := all

.PHONY: all help

all:
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest

help:
	@printf '%s\n' \
	  'Default: make (or make all) runs ruff check, ruff format --check, then pytest' \
	  'Assignment 6 focused branch — older A5 make targets live in archive/pre_a6_coursework/root_files/Makefile'
