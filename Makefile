.PHONY: add_all modal_app_list deploy test lint docs ask sweep rag

add_all:
	uv run ruff check --fix .
	uv run ruff format .
	git add .

test:
	uv run pytest

# ruff has no module-length rule, so wc -l enforces the 300-line limit
lint:
	uv run ruff check .
	uv run ruff format --check .
	wc -l src/*.py tests/*.py | awk '$$2 != "total" && $$1 > 300 {print $$2 " has " $$1 " lines (limit 300)"; bad=1} END {exit bad}'

# Modal needs the proxy's certificates: ~/.pythoncustomize/sitecustomize.py injects them
modal_app_list:
	PYTHONPATH=$$HOME/.pythoncustomize uv run modal app list

deploy:
	PYTHONPATH=$$HOME/.pythoncustomize uv run modal deploy -m src.modal_server

# generates (once) and prints the 10 supporting documents with token counts
docs:
	uv run --env-file .env python -m src.vectordb docs

# top-1 query on the first question at 3072 dimensions
ask:
	uv run --env-file .env python -m src.vectordb ask

# per-size check that every question's own document is the nearest one
sweep:
	uv run --env-file .env python -m src.vectordb sweep

# retrieve at DIM dimensions, answer with the base model on Modal, judge; writes results/rag.json
# usage: make rag DIM=128 (one of 3072 1536 768 256 128 64 32)
DIM ?= 64
rag:
	uv run --env-file .env python -m src.rag $(DIM)
