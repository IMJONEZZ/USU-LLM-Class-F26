.PHONY: add_all, run_modal, stop

add_all:
	uv run ruff check --fix .
	uv run ruff format .
	git add .

run_modal:
	PYTHONPATH=$$HOME/.pythoncustomize uv run modal run src/modal_server.py::$(FUNCTION)

stop:
	PYTHONPATH=$HOME/.pythoncustomize uv run modal app stop vllm-inference --yes
# PYTHONPATH=$HOME/.pythoncustomize uv run modal run src/modal_server.py::$(FUNCTION)