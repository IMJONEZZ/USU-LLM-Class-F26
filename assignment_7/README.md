# Assignment 7: serve a model

A vLLM server on Modal that serves Llama 3.2 1B Instruct and my Assignment 6 LoRA adapter (`a6`) side by side, protected by an API key. `src/inference.py` is the `requests` client for it.

| File | What it is |
|---|---|
| `assignment_7/modal_vllm.py` | The Modal app: `serve` is the deployed server, `test_lora` was a one-off check that vLLM can load the adapter on a T4 |
| `src/inference.py` | Client with a timeout, retries on timeouts and server errors, no retry on client errors, and the key hidden in printed output |
| `tests/test_inference.py` | Tests with mocked `requests`, so they need no server or key |
| `assignment_7/ten_question_check.py` | Sends the ten Assignment 6 questions to the base model and the adapter |
| `assignment_7/results/ten_question_check.json` | Raw answers, latencies and the summary from that check |

Run the client (the key is read from `VLLM_API_KEY`, and the first request after the server has been idle takes about 3.5 minutes):

    export VLLM_API_KEY=...        # not stored in this repo
    uv run python src/inference.py "How many Ls are in the word parallel?"

Results, screenshots and my reflection are in the comments on pull request #67. The server stays deployed until grading is finished, then I stop it and delete the volumes and the secret.
