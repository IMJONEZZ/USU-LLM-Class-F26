SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.DEFAULT_GOAL := all

export TMPDIR ?= $(HOME)/.cache/uv-tmp
export UV_CACHE_DIR ?= $(HOME)/.cache/uv

# Primary demo path = Colab GPU + Guidance (see Action Plan).
# Always request a GPU. Default T4 (fits 3B 4-bit on free tier); Pro can use A100.
GPU ?= T4
PROMPT ?=
MODEL_ID ?=
KEEP ?=

.PHONY: all help sync serve infer colab colab-keep

all:
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest

sync:
	mkdir -p "$(TMPDIR)" "$(UV_CACHE_DIR)"
	uv sync --group dev

serve:
	uv run uvicorn src.server:app --host 127.0.0.1 --port 8000 --workers 1

infer:
	uv run python -c "import requests; \
r=requests.post('http://127.0.0.1:8000/generate-html', json={'prompt':'Short page about USU'}); \
print(r.status_code, r.headers.get('content-type')); print(r.text)"

# Always passes --gpu (never CPU). Falls back T4→L4→A100→G4→H100 if needed.
colab:
	@args=(--gpu "$(GPU)"); \
	if [[ -n "$(PROMPT)" ]]; then args+=(--prompt "$(PROMPT)"); fi; \
	if [[ -n "$(MODEL_ID)" ]]; then args+=(--model-id "$(MODEL_ID)"); fi; \
	if [[ -n "$(KEEP)" ]]; then args+=(--keep); fi; \
	python3 scripts/run_colab_a9.py "$${args[@]}"

colab-keep:
	$(MAKE) colab KEEP=1

help:
	@printf '%s\n' \
	  'make / make all   ruff + pytest' \
	  'make sync         recreate .venv' \
	  'make serve        local FastAPI on :8000 (secondary)' \
	  'make infer        POST /generate-html (local server must be up)' \
	  'make colab        PRIMARY: Colab GPU (default T4) + Guidance + Qwen 1.5B 4-bit' \
	  'make colab GPU=A100 PROMPT="..." MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct' \
	  'make colab-keep   same as colab but leave the VM running'
