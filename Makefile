.PHONY: add_all

add_all:
	uv run ruff check --fix .
	uv run ruff format .
	git add .
