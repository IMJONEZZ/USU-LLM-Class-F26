QUESTION ?= Where did fortune cookies originate?

.PHONY: add_all modal_app_list deploy ask check stream test lint

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

# needs MODAL_ID, MODAL_SECRET and MODAL_URL in .env; make ask QUESTION="your question"
ask:
	uv run --env-file .env python -m src.inference ask "$(QUESTION)"

# the 8 test questions on the adapter and on the base; writes serving_runs/m4_correctness.json
check:
	uv run --env-file .env python -m src.inference check assets/test_set.jsonl

# streams one question and prints first-token time and tokens per second; if the server had to cold-start, streams again (second = warm); writes serving_runs/m5_stream.json
stream:
	uv run --env-file .env python -m src.inference stream "$(QUESTION)"
