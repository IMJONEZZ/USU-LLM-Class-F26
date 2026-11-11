.PHONY: add_all modal_app_list deploy models lm-demo nltk_data data fig-split checks test lint

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
	$(PY) src.plots

# build the image and deploy the vLLM server on a T4; it prints the URL to save in .env as MODAL_URL
deploy:
	PYTHONPATH=$$HOME/.pythoncustomize uv run modal deploy -m src.modal_server

# ask the server which models it serves, with and without Modal's proxy tokens (the second call should be refused)
models:
	uv run --env-file .env sh -c 'curl -s -w "\n[with tokens: HTTP %{http_code}]\n" -H "Modal-Key: $$MODAL_ID" -H "Modal-Secret: $$MODAL_SECRET" "$$MODAL_URL/v1/models"; curl -s -o /dev/null -w "[without tokens: HTTP %{http_code}]\n" "$$MODAL_URL/v1/models"'

# one real call (twice) through DSPy to the Modal server, then print the messages DSPy sent
lm-demo:
	$(PY) src.optimize

# Google's checker needs this sentence-splitter data once per machine (CI downloads it too)
nltk_data:
	PYTHONPATH=$$HOME/.pythoncustomize uv run python -m nltk.downloader punkt_tab

# Modal needs the proxy's certificates: ~/.pythoncustomize/sitecustomize.py injects them
modal_app_list:
	PYTHONPATH=$$HOME/.pythoncustomize uv run modal app list
