MODAL_FILE ?= scripts/modal_baseline.py

.PHONY: add_all modal_run modal_app_list modal_test

add_all:
	uv run ruff check --fix .
	uv run ruff format .
	git add .

modal_run:
	PYTHONPATH=$$HOME/.pythoncustomize uv run modal run $(MODAL_FILE) $(ARGS)

modal_app_list:
	PYTHONPATH=$$HOME/.pythoncustomize uv run modal app list

modal_test:
	PYTHONPATH=$$HOME/.pythoncustomize uv run modal run scripts/modal_tests.py
