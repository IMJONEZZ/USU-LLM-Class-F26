.PHONY: add_all, download_onto_volume, deploy_server, stop, trigger_start, check_logs, test

location := PYTHONPATH=$$HOME/.pythoncustomize uv

add_all:
	uv run ruff check --fix .
	uv run ruff format .
	git add .

download_onto_volume:
	$(location) run modal run src/modal_server.py::download_model

deploy_server:
	$(location) run modal deploy src/modal_server.py

trigger_start:
	$(location) run pytest tests/test_llm.py -s

stop:
	$(location) run modal app stop vllm-inference --yes

check_logs:
	$(location) run modal app logs vllm-inference

test:
	$(location) run pytest