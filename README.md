# Assignment 11 — Local Gradio Chatbot

A minimal multi-turn chatbot. A [Gradio](https://www.gradio.org/) page streams replies from a
Qwen model served by a local [Ollama](https://ollama.com/) instance. Everything runs on one
machine (NVIDIA Quadro T2000, 4 GB VRAM); there is no cloud inference, API key,
database, or authentication.

## Architecture

```
browser ──(SSH tunnel)──▶ Gradio (127.0.0.1:7860) ──ollama Python client──▶ Ollama (127.0.0.1:11434) ──▶ GPU
```

- `src/frontend.py` — the whole app: Gradio Blocks UI, history handling, streaming, error handling.
- `tests/test_frontend.py` — pytest suite with a mocked Ollama client (no server, GPU, or models needed).

| Model | Ollama tag | Notes (measured on this machine) |
|---|---|---|
| Qwen2.5-3B-Instruct (**default**) | `qwen2.5:3b-instruct-q4_K_M` | ~67 tokens/s, 100% GPU. No thinking mode. |
| Qwen3.5-4B | `qwen3.5:4b-q4_K_M` | ~17 tokens/s, 62% GPU / 38% CPU. Optional thinking mode. |

Requests use `num_ctx=4096`. Conversation history is kept in Gradio's per-session `gr.State` as
plain `{"role", "content"}` messages and is resent on every turn.

## Controls

- **Model dropdown** — selects which Ollama model receives requests. Changing it moves the
  current chat to **Earlier chats** (so responses from different models never mix), starts an
  empty chat, resets Thinking mode to off, and asks Ollama to unload the previously running
  model so only one model occupies the 4 GB GPU.
- **Thinking mode** — checkbox that sends Ollama's `think` option. Enabled only for Qwen3.5-4B
  (hint underneath: "Slower"; off by default, explicitly sent as `think=False`); greyed out for
  Qwen2.5 with the hint "Not supported". The model's
  reasoning is never shown, only its final answer. Expect much slower replies when it is on.
- **Message box** — type and press Enter (or the send arrow). Blank input is ignored. Replies stream in.
- **Per-reply stats** — each response shows its own generation speed and time in small text
  beside its copy button, for example `66.9 tokens/s (120 tokens) · 1.8 s total`. Copy copies only
  the reply text, not the stats. The speed is computed from
  the token count and generation time Ollama reports; the total is Ollama's `total_duration` for
  that request (it includes loading the model if it was not already in memory, so the first reply
  after a model switch is slower). Either part is left out if Ollama does not report it. The stats
  stay with their response, including when an earlier chat is reopened, and are not sent back to
  Ollama. With Thinking mode on, the token count includes the hidden reasoning tokens.
- **Header** — shows where the app is running: `Assignment 11 - Running Locally` with the GPU
  (from `nvidia-smi`, e.g. `Quadro T2000 (4 GB)`), or
  `Assignment 11 - Running on Colab` with the Colab GPU. Detection uses Colab's environment, and
  shows "No GPU detected" when `nvidia-smi` is unavailable.
- **Theme** — follows the browser's light/dark setting (or add `?__theme=dark` to the URL). Dark
  mode is true black for OLED screens, with neutral grey accents.

## Setup

Requirements: Python 3.14, [uv](https://docs.astral.sh/uv/), and Ollama.

```bash
uv sync                       # or: make sync
```

Python dependencies (all installed by `uv sync`): `gradio`, `ollama`, plus `pytest`, `pytest-cov`,
and `ruff` for the tests and lint checks.

### Ollama

Ollama is already running on the lab machine. To check it, or start/pull manually:

```bash
curl http://127.0.0.1:11434/api/tags        # lists installed models
ollama serve                                # start the server if it is not running
ollama pull qwen2.5:3b-instruct-q4_K_M      # one-time downloads (the app never pulls models)
ollama pull qwen3.5:4b-q4_K_M
```

## Run the app

On the lab machine:

```bash
make run                      # same as: uv run python -m src.frontend
```

It binds to `127.0.0.1:7860` only. To use it from your own computer, forward the port over SSH:

```bash
ssh -L 7860:127.0.0.1:7860 <user>@<lab-server-address>
```

then open <http://127.0.0.1:7860> in your local browser.

Optional environment variables: `OLLAMA_HOST` (default `http://127.0.0.1:11434`),
`GRADIO_SERVER_NAME` (default `127.0.0.1`), `GRADIO_SERVER_PORT` (default `7860`).

## Tests and linting

```bash
make test       # uv run pytest   (coverage of src/ must be 100%)
make lint       # ruff check + ruff format --check
make format     # apply ruff fixes and formatting
make            # lint + test, same as CI
```
