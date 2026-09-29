FILE ?= src/main.py

.PHONY: add_all gpu_start gpu_add_dependencies gpu_run gpu_stop gpu_test modal_run modal_app_list modal_test

add_all:
	uv run ruff check --fix .
	uv run ruff format .
	git add .

gpu_start:
	PYTHONPATH=$$HOME/.pythoncustomize uv run colab new -s usu-gpu --gpu T4
gpu_add_dependencies:
	PYTHONPATH=$$HOME/.pythoncustomize uv run colab install -s usu-gpu -r pyproject.toml
gpu_run:
	PYTHONPATH=$$HOME/.pythoncustomize uv run colab exec -s usu-gpu -f $(FILE)
gpu_stop:
	PYTHONPATH=$$HOME/.pythoncustomize uv run colab stop -s usu-gpu
gpu_test:
	echo "!uv run pytest" | PYTHONPATH=$$HOME/.pythoncustomize uv run colab exec -s usu-gpu


MODAL_FILE ?= src/gpu_check.py

modal_run:
	PYTHONPATH=$$HOME/.pythoncustomize uv run modal run $(MODAL_FILE)
modal_app_list:
	PYTHONPATH=$$HOME/.pythoncustomize uv run modal app list

modal_test:
	PYTHONPATH=$$HOME/.pythoncustomize uv run modal run src/generation.py::run_tests
