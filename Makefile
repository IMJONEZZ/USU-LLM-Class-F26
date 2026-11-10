.PHONY: add_all modal_app_list nltk_data data fig-split test lint

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
	PYTHONPATH=$$HOME/.pythoncustomize uv run python -m src.data

# stacked-bar chart of prompts per type in each split, saved to results/figures/split.png
fig-split:
	PYTHONPATH=$$HOME/.pythoncustomize uv run python -m src.plots

# Google's checker needs this sentence-splitter data once per machine (CI downloads it too)
nltk_data:
	PYTHONPATH=$$HOME/.pythoncustomize uv run python -m nltk.downloader punkt_tab

# Modal needs the proxy's certificates: ~/.pythoncustomize/sitecustomize.py injects them
modal_app_list:
	PYTHONPATH=$$HOME/.pythoncustomize uv run modal app list
