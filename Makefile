MODAL_FILE ?= scripts/modal_app.py
ENTRY ?= eval_split

.PHONY: add_all modal_run modal_app_list

add_all:
	uv run ruff check --fix .
	uv run ruff format .
	git add .

# Modal needs the proxy's certificates: ~/.pythoncustomize/sitecustomize.py injects them
modal_run:
	PYTHONPATH=$$HOME/.pythoncustomize uv run modal run $(MODAL_FILE)::$(ENTRY) $(ARGS)

modal_app_list:
	PYTHONPATH=$$HOME/.pythoncustomize uv run modal app list
