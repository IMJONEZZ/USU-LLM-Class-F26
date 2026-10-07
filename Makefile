SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.DEFAULT_GOAL := help

DOCKER ?= sudo --preserve-env=HF_TOKEN docker
RUN ?= docker-run
TRAIN ?= train
EPOCHS ?= 3
A5_IMAGE ?= usu-llm-training:a5
A5_OUTPUT_DIR ?= $(CURDIR)/data/assignment5/$(RUN)
A5_CACHE_DIR ?= $(CURDIR)/.cache/assignment5-docker
A5_CORPUS ?= $(CURDIR)/SW_EpisodeIV_VI.json
A5_SPLITS_DIR ?= $(CURDIR)/data/evaluation
export A5_IMAGE A5_OUTPUT_DIR A5_CACHE_DIR A5_CORPUS A5_SPLITS_DIR TRAIN EPOCHS

.PHONY: help a5-build a5-test a5-gpu a5-prepare a5-feasibility a5-baseline a5-train a5-evaluate a5-report a5-run

help:
	@printf '%s\n' \
	  'Assignment 5 Docker workflow (run from the repository root):' \
	  '  make a5-build        Build the pinned training image' \
	  '  make a5-test         Run container CPU tests; no GPU required' \
	  '  make a5-gpu          Check GPU visibility' \
	  '  make a5-prepare      Prepare data using the existing split manifest' \
	  '  make a5-feasibility  Check real GPU optimizer updates and memory' \
	  '  make a5-baseline     Evaluate the original model' \
	  '  make a5-train        Train and select the best development checkpoint' \
	  '  make a5-evaluate     Evaluate the selected adapter' \
	  '  make a5-report       Generate the before/after report' \
	  '  make a5-run RUN=new  Run every stage in order in a fresh directory' \
	  '' \
	  'Options: RUN=docker-run TRAIN=train EPOCHS=3 DOCKER=docker' \
	  'Use TRAIN=train-retry1 for the completed experiment. Existing outputs are preserved.'

a5-build:
	@$(DOCKER) build --platform linux/amd64 -f Dockerfile.training -t "$$A5_IMAGE" .

a5-test:
	@$(DOCKER) run --rm --user "$$(id -u):$$(id -g)" "$$A5_IMAGE" \
	  -m pytest -c /app/pyproject.toml --no-cov -p no:cacheprovider -q \
	  /app/tests/test_trainer.py /app/tests/test_llama_evaluator.py \
	  /app/tests/test_training_report.py /app/tests/test_trainer_gpu.py

# One wrapper keeps mounts, authentication and exit-code handling identical.
a5-gpu a5-prepare a5-feasibility a5-baseline a5-train a5-evaluate a5-report:
	@auth=(-e HF_TOKEN)
	token_file="$${HF_TOKEN_PATH:-$${HF_HOME:-$$HOME/.cache/huggingface}/token}"
	if [[ -f "$$token_file" ]]; then
	  auth+=(--mount "type=bind,src=$$token_file,dst=/run/secrets/hf_token,readonly" -e HF_TOKEN_PATH=/run/secrets/hf_token)
	fi
	gpu=(--gpus all)
	inputs=()
	required=(prepared/prepared.json)
	artifact=""
	log="$(@:a5-%=%).log"
	case "$@" in
	  a5-gpu)
	    required=()
	    command=(-c 'from src.trainer import gpu_environment; print(gpu_environment())') ;;
	  a5-prepare)
	    gpu=(); required=(); artifact=prepared
	    [[ -f "$$A5_CORPUS" && -f "$$A5_SPLITS_DIR/splits.json" ]] || { printf '%s\n' 'Corpus or existing splits.json is missing.' >&2; exit 1; }
	    inputs=(--mount "type=bind,src=$$A5_CORPUS,dst=/inputs/corpus.json,readonly" --mount "type=bind,src=$$A5_SPLITS_DIR,dst=/inputs/evaluation,readonly")
	    command=(-m src.trainer prepare --corpus /inputs/corpus.json --splits /inputs/evaluation/splits.json --output /outputs/prepared) ;;
	  a5-feasibility)
	    artifact=feasibility
	    command=(-m src.trainer feasibility --prepared /outputs/prepared --output /outputs/feasibility) ;;
	  a5-baseline)
	    artifact=before.json
	    command=(-m src.evaluator llama --prepared /outputs/prepared --output /outputs/before.json) ;;
	  a5-train)
	    required+=(feasibility/success.json before.json)
	    artifact="$$TRAIN"; log="$$TRAIN.log"
	    command=(-m src.trainer train --prepared /outputs/prepared --feasibility /outputs/feasibility --epochs "$$EPOCHS" --output "/outputs/$$TRAIN") ;;
	  a5-evaluate)
	    required+=("$$TRAIN/success.json" "$$TRAIN/best/adapter_model.safetensors" before.json)
	    artifact=after.json
	    command=(-m src.evaluator llama --prepared /outputs/prepared --adapter "/outputs/$$TRAIN/best" --output /outputs/after.json) ;;
	  a5-report)
	    gpu=(); required=(before.json after.json "$$TRAIN/training.json"); artifact=results.md
	    command=(-m src.training_report --before /outputs/before.json --after /outputs/after.json --training "/outputs/$$TRAIN/training.json" --output /outputs/results.md) ;;
	esac
	if [[ -n "$$artifact" && -e "$$A5_OUTPUT_DIR/$$artifact" ]]; then
	  printf 'Output already exists: %s. Use a fresh RUN, or a fresh TRAIN for a training retry.\n' "$$A5_OUTPUT_DIR/$$artifact" >&2
	  exit 1
	fi
	for file in "$${required[@]}"; do
	  [[ -f "$$A5_OUTPUT_DIR/$$file" ]] || { printf 'Missing prerequisite: %s\n' "$$A5_OUTPUT_DIR/$$file" >&2; exit 1; }
	done
	mkdir -p "$$A5_OUTPUT_DIR" "$$A5_CACHE_DIR"
	$(DOCKER) run --rm "$${gpu[@]}" --user "$$(id -u):$$(id -g)" \
	  --mount "type=bind,src=$$A5_OUTPUT_DIR,dst=/outputs" \
	  --mount "type=bind,src=$$A5_CACHE_DIR,dst=/tmp/a5-cache" \
	  -e MPLCONFIGDIR=/tmp/a5-cache/matplotlib \
	  "$${inputs[@]}" "$${auth[@]}" "$$A5_IMAGE" "$${command[@]}" \
	  2>&1 | tee -a "$$A5_OUTPUT_DIR/$$log"

# Recursive calls are deliberately sequential, even when invoked with make -j.
a5-run:
	@if [[ -e "$$A5_OUTPUT_DIR" ]]; then
	  printf 'Run directory already exists: %s. Choose a new RUN name.\n' "$$A5_OUTPUT_DIR" >&2
	  exit 1
	fi
	for target in a5-build a5-test a5-gpu a5-prepare a5-feasibility a5-baseline a5-train a5-evaluate a5-report; do
	  $(MAKE) --no-print-directory "$$target"
	done
