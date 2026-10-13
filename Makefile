MODAL_FILE ?= scripts/modal_app.py
ENTRY ?= eval_split
ROUND ?= 0
SPLIT ?= eval
OUTPUTS ?= data/round$(ROUND)_eval_outputs.jsonl
OUT ?= results/round$(ROUND)_judged.json

.PHONY: add_all modal_run modal_app_list test lint baseline probe train eval_finetuned final_test judge decide

add_all:
	uv run ruff check --fix .
	uv run ruff format .
	git add .

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

# Modal needs the proxy's certificates: ~/.pythoncustomize/sitecustomize.py injects them
modal_run:
	PYTHONPATH=$$HOME/.pythoncustomize uv run modal run $(MODAL_FILE)::$(ENTRY) $(ARGS)

modal_app_list:
	PYTHONPATH=$$HOME/.pythoncustomize uv run modal app list

# zero-shot baseline on the eval split
baseline:
	$(MAKE) --no-print-directory modal_run ARGS="--split eval --prompt-style baseline"

# 2-epoch run on 8 rows, saves nothing
probe:
	$(MAKE) --no-print-directory modal_run ENTRY=train_round ARGS="--probe"

# round 0 training; later rounds: make modal_run ENTRY=train_round ARGS="--round-n 1 ..."
train:
	$(MAKE) --no-print-directory modal_run ENTRY=train_round

# make eval_finetuned ROUND=0 -> results/round0_eval.json
eval_finetuned:
	$(MAKE) --no-print-directory modal_run ARGS="--split eval --prompt-style finetuned --adapter-path /checkpoints/round_$(ROUND) --output results/round$(ROUND)_eval.json"

# the final test set is run once; a second run needs --force
final_test:
	$(MAKE) --no-print-directory modal_run ARGS="--split test --prompt-style finetuned --adapter-path /checkpoints/round_$(ROUND)"

# Gemini judges the saved outputs; the key comes from .env
judge:
	uv run --env-file .env python scripts/judge_outputs.py --split $(SPLIT) --outputs $(OUTPUTS) --out $(OUT)

# weak categories and continue/stop from the judged scores: make decide ROUND=0 GAP=0.3 [DROP=0.1]
decide:
	uv run python scripts/decide_round.py --round $(ROUND) --gap $(GAP) $(if $(DROP),--drop $(DROP))
