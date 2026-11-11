# Assignment 10: prompt optimization

Optimizing a prompt with DSPy, without constrained sampling. The task is letter counting ("How many times does the letter 'l' appear in the word 'parallel'?") with Llama 3.2 3B run by Ollama on a Mac. I compared the plain prompt with two optimizers (BootstrapFewShot and GEPA) on the same 60 test questions.

| File | What it is |
|---|---|
| `src/prompt_optimization.py` | The questions and word split, the answer checker, the baseline program, both optimizers, the experiment, the demo and the command line |
| `tests/test_prompt_optimization.py` | Tests, using DSPy's fake model; one real-model test runs only with `RUN_REAL_MODEL=1` |
| `assignment_10/results/optimization_results.json` | Run 1: BootstrapFewShot, every answer before and after |
| `assignment_10/results/gepa_results.json` | Run 2: GEPA, every answer before and after, and the new instruction |
| `assignment_10/screenshots/` | Two before and after queries, and the tests and ruff passing |

Run it (needs `ollama serve` and `ollama pull llama3.2`):

    uv run python -m src.prompt_optimization run
    uv run python -m src.prompt_optimization run-gepa
    uv run python -m src.prompt_optimization demo "How many times does the letter 'a' appear in the word 'accommodate'?"
