# USU LLM Class — Fall 2026

Homework-submission repository for USU LLM Class DSAI-5810/6810.

## Setup

Requires Python 3.14+.

```bash
uv sync --group dev
```

Checks: `make` (ruff + pytest).

## Progress

- A0–A8 — done (prior branches)
- A9 — HTML Structured Output (in progress)

## Assignment 9: Constrained HTML output

**Primary path (Action Plan):** Colab Pro GPU + [Guidance](https://github.com/guidance-ai/guidance)
+ `Qwen/Qwen2.5-1.5B-Instruct`. Constrained HTML grammar during decoding
(`gen(max_tokens=...)` / `repeat(max=...)`); then `html_grammar().match(...)`
+ whitelist validation. API returns raw `text/html`.

### Run on Colab (default / primary)

Needs [`colab`](https://pypi.org/project/google-colab-cli/) CLI authenticated.
`make colab` **always** requests a GPU (never CPU). Default is **T4** — smallest
Colab GPU that safely runs `Qwen2.5-1.5B-Instruct` in 4-bit (a few GB VRAM; T4 has
16GB), and free-tier accounts can get it. If T4 is unavailable it walks
T4 → L4 → A100 → G4 → H100.

```bash
make colab
# Pro / bigger card if you want:
make colab GPU=A100 PROMPT='Short page about Aggie Ice Cream'
```

Output: `artifacts/colab/a9_output.html` — screenshot that for the PR.

### Local secondary (optional)

```bash
A9_LOAD_IN_4BIT=true make serve
make infer   # or open http://127.0.0.1:8000/
```

Env knobs: `A9_MODEL_ID`, `A9_LOAD_IN_4BIT`, `A9_COLAB_GPU`.

CI uses mocked generators in `tests/test_html.py`.

Allowed tags: `html`, `head`, `title`, `body`, `h1`, `h2`, `p`, `ul`, `ol`,
`li`, `strong`, `em`, `br`. No attributes.

## AI assistance

Claude Code and Codex follow [AGENTS.md](AGENTS.md).
