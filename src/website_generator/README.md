# Grammar-constrained AI website generator

A local FastAPI app and vanilla browser frontend backed by Qwen2.5-3B-Instruct
served by vLLM with XGrammar. No model calls go to a hosted service.

## Status

Backend tests use mocked inference; they do not establish GPU compatibility.
GPU inference and a real submission screenshot have **not** been verified in
this coding session: `nvidia-smi` reported `GPU access blocked by the operating system`.
Follow the steps below in your own WSL2 terminal to finish those acceptance checks.

## 1. WSL2 and GPU setup

In **Windows PowerShell**, check or update WSL (install Ubuntu if needed):

```powershell
wsl --status
wsl --update
wsl --install -d Ubuntu
```

Install a current NVIDIA **Windows** driver supporting the RTX 5080. Restart
Windows if required. Do not install a Linux display driver inside WSL.
In Ubuntu, check `nvidia-smi`; it must identify your GPU before continuing.

Use the **repository-level uv environment** for both services. Run all commands
below from the repository root; there are no project-specific virtual environments
or requirements files. The repo's Python version is selected by `.python-version`.

```bash
uv sync --group inference
uv run --group inference python -c 'import torch, vllm; print("torch", torch.__version__, "CUDA", torch.version.cuda, "vLLM", vllm.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(), torch.cuda.get_device_capability()); print(torch.cuda.get_arch_list()); print((torch.ones(8, device="cuda") + 1).cpu())'
uv run --group inference python -m src.website_generator.scripts.check_grammar
```

The `inference` dependency group installs vLLM on Linux/WSL. Ordinary `uv sync`
installs the backend and development tools without downloading the GPU stack.
Use `--group inference` for **both** servers so uv keeps the inference packages
installed in the shared `.venv`.

The [official vLLM GPU installation guide](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
requires CUDA 12.8 or newer for Blackwell. Verify the printed CUDA version is at least 12.8 and the
reported GPU is the RTX 5080 (compute capability 12.0). An actual tensor operation
and then model generation are required; a CUDA version string alone is insufficient.
If CUDA is unavailable, too old, or reports an unsupported architecture, update
the Windows driver and reinstall a compatible vLLM/PyTorch wheel combination
following that guide. Do not independently overwrite vLLM's pinned torch build.
See also [NVIDIA's CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

The grammar check compiles the actual file using XGrammar and exercises valid,
invalid and incomplete fragments without downloading a model. Stop if it fails.
Then start inference (first launch downloads the model):

```bash
bash src/website_generator/scripts/start_vllm.sh
```

The script binds **127.0.0.1:8000**, enables XGrammar, caps context at 4096 tokens,
uses eager execution, one concurrent sequence and 80% GPU memory. These are starting settings for
16 GB VRAM, not a verified memory benchmark. Close other GPU workloads if needed.

## 2. Start the backend and frontend

In a second WSL terminal, from the repository root:

```bash
uv run --group inference uvicorn src.website_generator.backend.app:app --host 127.0.0.1 --port 8001
```

Open **http://localhost:8001** in your Windows browser. FastAPI serves both the
frontend and API; no CORS configuration or file opening is needed. If Windows
localhost forwarding fails, check your WSL configuration; keep inference bound
to loopback. Do not expose vLLM to your LAN.

Describe a small page and click **Generate webpage**. Preview and source appear
side by side. The browser cannot provide its own grammar or model settings.

```bash
curl --fail-with-body http://127.0.0.1:8001/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A neighborhood book club with a welcome heading and three upcoming meetings."}'
```

Expected shape: `{"html":"<section>...</section>"}`. For a minimal real generation
check first try `{"prompt":"One paragraph saying Hello."}`. A 502/504 is a failed
check, not a successful demonstration. If you change models, export the same
`VLLM_MODEL` in both terminals before launching either service.

## Grammar and validation

`backend/grammar.gbnf` uses XGrammar's EBNF syntax despite the conventional `.gbnf`
filename. Every HTTP inference request includes `structured_outputs.grammar`, as
specified in the [vLLM structured output documentation](https://docs.vllm.ai/en/latest/features/structured_outputs/).
This constrains decoding, rather than relying on prompt instructions.

Supported roots: div, section, article, h1–h3, p and ul. Containers nest; ul holds
one or more li elements. Headings and paragraphs permit strong, em and links.
Links contain text and an HTTPS URL with a domain and optional simple path;
queries, fragments, credentials, ports and other attributes are excluded.
Text supports Unicode and five explicit HTML entities. Scripts, styles,
event handlers and arbitrary attributes have no grammar production.

The backend rejects unbalanced, unexpected or noncanonical HTML and validates
link hostnames, then serializes the allowlisted fragment. It also limits nesting
and response size. This semantic validation deliberately rejects some strings
permitted by the grammar (such as malformed domain labels). It never silently
repairs truncated output. Only a `stop` finish reason is accepted. Errors return
422 for invalid requests, 502 for inference/HTML failures and 504 for timeouts.

The preview has an empty iframe sandbox and a restrictive CSP. HTML is assigned
only to the iframe's srcdoc; raw source uses textContent. Preview CSS is fixed
application code, never model-provided. Network resources and scripts are blocked.

## Tests

From the **repository root** after `uv sync`:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

GPU-independent tests mock the vLLM HTTP response and check grammar attachment,
request validation, safe HTML, malformed HTML, failed inference and truncation.
For the real grammar smoke check, run:

```bash
uv run --group inference python -m src.website_generator.scripts.check_grammar
```

## Submission screenshot (manual, real inference required)

1. Start both servers, open the page in Windows and enter a book-club prompt.
2. Generate successfully and ensure the submitted prompt, returned source and
   rendered preview are all visible (zoom out if needed).
3. Use Windows Snipping Tool to capture the page and save it as
   `screenshots/generation.png` at the repository root.
4. Review the image and include it in the assignment PR.

Screenshot artifact location: `screenshots/generation.png` (pending a real run).
No placeholder or mocked screenshot is presented as submission evidence.
After capture, embed it with `![Real local generation](../../screenshots/generation.png)`.
Record the installed inference versions for reproducibility:

```bash
uv pip freeze --python .venv/bin/python > screenshots/inference-environment.txt
```

## Files

- `src/website_generator/backend/`: FastAPI endpoint, validator and grammar.
- `src/website_generator/frontend/`: HTML/CSS/JavaScript UI served by FastAPI.
- `src/website_generator/scripts/`: local inference launcher and GPU-independent grammar smoke check.
- `tests/test_website_generator.py`: mocked API and validation tests.
- `screenshots/`: real submission evidence to capture after inference succeeds.

## Startup compilation failure

The launcher uses `--enforce-eager` to bypass torch.compile and CUDA graphs,
following the [vLLM compilation debugging guide](https://docs.vllm.ai/en/latest/design/debug_vllm_compile/).
This avoids the TorchInductor startup compilation path that failed in the supplied
WSL log. XGrammar constrained decoding remains enabled; throughput may be lower.
This is a workaround, not verification that the CUDA toolchain is repaired.

If a traceback ends with `PermissionError: [Errno 13] Permission denied: 'nvcc'`
inside `_cuda_system_info_comment`, PyTorch failed while collecting compiler
error diagnostics. That final exception can mask the original compilation issue.
For this assignment, retry the updated launcher first:

```bash
bash src/website_generator/scripts/start_vllm.sh
```

If it still fails, retain the complete startup log, including lines before the
traceback. To investigate the toolchain separately, run these in your WSL terminal:

```bash
command -v nvcc
which -a nvcc
nvcc --version
```

The compiler must be a Linux executable accessible to WSL. The supplied log alone
does not establish which PATH entry or permission caused the failure. Eager mode
does not eliminate every possible CUDA kernel compilation requirement.

### FlashInfer sampling cannot find nvcc

If startup reaches `_dummy_sampler_run` and fails in
`flashinfer.sampling.get_sampling_module` with
`Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`,
FlashInfer is attempting to compile its sampling kernel without a discoverable
CUDA compiler toolkit. Eager execution alone does not bypass this separate JIT.

The launcher defaults `VLLM_USE_FLASHINFER_SAMPLER=0`, selecting vLLM's
PyTorch-native sampling fallback. This avoids that specific FlashInfer sampling
build; XGrammar remains enabled. Restart with:

```bash
bash src/website_generator/scripts/start_vllm.sh
```

If you previously exported `VLLM_USE_FLASHINFER_SAMPLER=1`, override it explicitly:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 bash src/website_generator/scripts/start_vllm.sh
```

The fallback is confirmed in the installed vLLM source and in the
[upstream sampler implementation](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py).
It does not repair or install nvcc and does not guarantee every other CUDA kernel
can run without a toolkit. GPU startup with this change remains to be verified.
