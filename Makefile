.PHONY: add_all, run_modal

add_all:
	uv run ruff check --fix .
	uv run ruff format .
	git add .

run_modal:
	PYTHONPATH=$$HOME/.pythoncustomize uv run modal run src/modal_server.py::$(FUNCTION)