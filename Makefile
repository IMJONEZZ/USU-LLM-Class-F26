.PHONY: add_all warm baseline gepa-smoke gepa fig-gepa smoke-record fig-gepa-smoke test-draws fig-draws greedy summary fig-summary modal_app_list deploy models lm-demo pilot-smoke pilot nltk_data data fig-split fig-pilot checks test lint

# every python target needs the proxy certificates, Google's checker folder on the import path, and the keys in .env
PY = PYTHONPATH=$$HOME/.pythoncustomize:third_party uv run --env-file .env python -m

add_all:
	uv run ruff check --fix .
	uv run ruff format .
	git add .

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

# download IFEval, filter, split, save assets/splits.json and print the count tables
data:
	$(PY) src.data

# run Google's strict check on all 300 prompts and print the feedback samples
checks:
	$(PY) src.checks

# stacked-bar chart of prompts per type in each split, saved to results/figures/split.png
fig-split:
	$(PY) src.plots split

# build the image and deploy the vLLM server on a T4; it prints the URL to save in .env as MODAL_URL
deploy:
	PYTHONPATH=$$HOME/.pythoncustomize uv run modal deploy -m src.modal_server

# ask the server which models it serves, with and without Modal's proxy tokens (the second call should be refused)
models:
	uv run --env-file .env sh -c 'curl -s -w "\n[with tokens: HTTP %{http_code}]\n" -H "Modal-Key: $$MODAL_ID" -H "Modal-Secret: $$MODAL_SECRET" "$$MODAL_URL/v1/models"; curl -s -o /dev/null -w "[without tokens: HTTP %{http_code}]\n" "$$MODAL_URL/v1/models"'

# one real call (twice) through DSPy to the Modal server, then print the messages DSPy sent
lm-demo:
	$(PY) src.optimize

# 3 tasks x 5 samples: read the raw replies by hand before the full pilot
pilot-smoke:
	$(PY) src.evaluate smoke

# the pilot: every train and validation task x 5 samples; refuses to overwrite an earlier results file
pilot:
	$(PY) src.evaluate pilot

# Google's checker needs this sentence-splitter data once per machine (CI downloads it too)
nltk_data:
	PYTHONPATH=$$HOME/.pythoncustomize uv run python -m nltk.downloader punkt_tab

# Modal needs the proxy's certificates: ~/.pythoncustomize/sitecustomize.py injects them
modal_app_list:
	PYTHONPATH=$$HOME/.pythoncustomize uv run modal app list

# histogram of the pilot's task pass rates, saved to results/figures/pilot.png (run after make pilot)
fig-pilot:
	$(PY) src.plots pilot

# wait until the Modal server answers (it scales to zero after 20 idle minutes and takes about 4 minutes to boot)
warm:
	uv run --env-file .env sh -c 'until [ "$$(curl -s -o /dev/null -w "%{http_code}" -H "Modal-Key: $$MODAL_ID" -H "Modal-Secret: $$MODAL_SECRET" "$$MODAL_URL/v1/models")" = 200 ]; do echo "waiting for the server"; sleep 15; done; echo "server ready"'

# Gemini writes the baseline instruction from one train task per type; saved once to assets/prompts.json
baseline:
	$(PY) src.optimize baseline

# a tiny GEPA run (6 train tasks, 40 metric calls) to watch one round; needs the server warm (make warm)
gepa-smoke:
	$(PY) src.optimize gepa-smoke

# the GEPA run (light budget, Gemini reflects); saves the instruction in assets/prompts.json and results/gepa.json
gepa:
	$(PY) src.optimize gepa

# chart of each GEPA candidate's validation score and what the run cost, saved to results/figures/gepa.png
fig-gepa:
	$(PY) src.plots gepa

# rebuild the smoke run's record from its log (it predates saved records), then draw it like the full run
smoke-record:
	$(PY) src.optimize smoke-record

fig-gepa-smoke:
	$(PY) src.plots gepa-smoke

# the M8 test run: 3 instructions x 75 test tasks x 20 draws, resumable; saves results/test_draws.jsonl
test-draws:
	$(PY) src.evaluate test-draws

# heatmap of pass rate by instruction type and instruction, saved to results/figures/draws.png
fig-draws:
	$(PY) src.plots draws

# one temperature-0 answer per test task and instruction, with average logprobs, plus a 30-answer rerun check; needs the server warm
greedy:
	$(PY) src.evaluate greedy

# the final comparison: paired bootstrap intervals, per-type table, rank correlation, screenshot task; saves results/summary.json
summary:
	$(PY) src.evaluate summary

# forest plot of GEPA's gain with both intervals, and confidence against pass rate, saved to results/figures/summary.png
fig-summary:
	$(PY) src.plots summary
