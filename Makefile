QUESTION ?= Where did fortune cookies originate?
RUN ?= coldstart

.PHONY: add_all modal_app_list deploy ask check stream coldstart experiments run5 chart test lint

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

# needs MODAL_ID, MODAL_SECRET and MODAL_URL in .env; make ask QUESTION="your question"
ask:
	uv run --env-file .env python -m src.inference ask "$(QUESTION)"

# the 8 test questions on the adapter and on the base; writes serving_runs/m4_correctness.json
check:
	uv run --env-file .env python -m src.inference check assets/test_set.jsonl

# streams one question and prints first-token time and tokens per second; if the server had to cold-start (labelled "cold start"), streams again (labelled "warm"); writes serving_runs/m5_stream.json
stream:
	uv run --env-file .env python -m src.inference stream "$(QUESTION)"

# one true cold start by stage after waiting for the server to scale to zero; writes serving_runs/coldstart.json
coldstart:
	PYTHONPATH=$$HOME/.pythoncustomize uv run --env-file .env python -m src.stages "$(QUESTION)" $(RUN)

# ALL FIVE M8 runs in order, then the chart. It first EMPTIES both cache Volumes (the model re-downloads), redeploys with FAST_BOOT on/off, and ends with FAST_BOOT on again. About 30-50 minutes.
experiments:
	PYTHONPATH=$$HOME/.pythoncustomize uv run --env-file .env python -m src.experiments all "$(QUESTION)"

# ONLY run 5 (FAST_BOOT on, model and compile files cached), on the state runs 1-4 left behind, then redraws the five-run chart. Refuses if runs 1-4 are not saved or the caches are not populated.
run5:
	PYTHONPATH=$$HOME/.pythoncustomize uv run --env-file .env python -m src.experiments run5 "$(QUESTION)"

# redraw the chart from the five saved runs without running anything on Modal
chart:
	uv run python -m src.chart m8_run1 m8_run2 m8_run3 m8_run4 m8_run5
