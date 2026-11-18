# Assignment 11: a chat frontend

A Gradio chat window for the letter counter from Assignment 10 (Llama 3.2 3B run by Ollama on a Mac). A radio switch under the chat box picks which prompt answers: the plain prompt, or the prompt GEPA wrote in Assignment 10. I sent the same 60 test questions through the window's reply function with each prompt: plain 20.0%, GEPA 41.7%.

| File | What it is |
|---|---|
| `src/frontend.py` | The reply function, the Gradio chat window with the radio switch, the scoring and pass bar, and the command line |
| `tests/test_frontend.py` | Tests, using a fake model; one test starts the real Gradio app and calls it through Gradio's test client |
| `assignment_11/results/frontend_results.json` | Every answer for both prompts on the 60 questions, and the pass bar check |
| `assignment_11/screenshots/` | The chat window with the plain prompt, the chat window with the GEPA prompt, and the tests and ruff passing |

This branch starts from my Assignment 10 branch, so the Assignment 10 files are here too.

Run it (needs `ollama serve` and `ollama pull llama3.2`):

    uv run python -m src.frontend serve
    uv run python -m src.frontend measure
