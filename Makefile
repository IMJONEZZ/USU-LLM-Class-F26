.PHONY: add_all modal_app_list deploy test lint smoke benchmark fetch summary

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

# 2 prompts through both methods (L4 first, then A10 if none is free); saves to the Modal results volume as benchmark_smoke.json (fetch it with: make fetch NAME=smoke)
smoke:
	PYTHONPATH=$$HOME/.pythoncustomize uv run modal run --detach -m src.modal_server::smoke

# the benchmark: all prompts through both methods (L4 first, then A10 if none is free), run inside the container with --detach so a sleeping
# laptop cannot stop it; saved to the Modal results volume; usage: make benchmark NAME=run1 (never overwrites); add COUNT=3 for only the first 3 prompts
NAME ?= run1
benchmark:
	PYTHONPATH=$$HOME/.pythoncustomize uv run modal run --detach -m src.modal_server::run --name $(NAME)$(if $(COUNT), --count $(COUNT))

# copy a finished run from the Modal results volume to results/ (the volume name must match config.RESULTS_VOLUME)
fetch:
	PYTHONPATH=$$HOME/.pythoncustomize uv run modal volume get --force webpage-benchmark-results benchmark_$(NAME).json results/

# summary table, summary JSON and the rendered gallery for a finished run: make summary NAME=run1
summary:
	uv run python -m src.benchmark results/benchmark_$(NAME).json
