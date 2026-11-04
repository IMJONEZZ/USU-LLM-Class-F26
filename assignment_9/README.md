# Assignment 9: HTML structured output

A web service that returns an HTML page for a prompt and forces the page to follow a grammar (Outlines, Qwen2.5-1.5B-Instruct on the Apple GPU). I compared it with the same model without the grammar on 10 prompts.

| File | What it is |
|---|---|
| `src/html_service.py` | The grammar, the page checker (strict and lenient), page generation, the comparison and the command line |
| `src/app.py` | The FastAPI service: `POST /generate` and `GET /health` |
| `tests/test_html.py` | Tests, using a fake model; one real-model test runs only with `RUN_REAL_MODEL=1` |
| `assignment_9/results/eval_results.json` | The 20 generated pages, with validity and seconds |
| `assignment_9/results/lenient_rescore.json` | The fairer re-score of the same pages |
| `assignment_9/screenshots/` | The service request and response, and the tests and ruff passing |

Run it:

    uv sync --group model
    uv run python -m src.html_service eval
    uv run uvicorn src.app:create_default_app --factory
