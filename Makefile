.PHONY: add_all modal_app_list deploy test lint

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
