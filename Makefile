.PHONY: add_all modal_app_list nltk_data test lint

add_all:
	uv run ruff check --fix .
	uv run ruff format .
	git add .

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

# Google's checker needs this sentence-splitter data once per machine (CI downloads it too)
nltk_data:
	PYTHONPATH=$$HOME/.pythoncustomize uv run python -m nltk.downloader punkt_tab

# Modal needs the proxy's certificates: ~/.pythoncustomize/sitecustomize.py injects them
modal_app_list:
	PYTHONPATH=$$HOME/.pythoncustomize uv run modal app list
