# Transformers/PEFT server

This folder keeps the original Transformers/PEFT model loader and the reusable
FastAPI request/response handling. It still loads the final Assignment 6 adapter
and supports CPU or CUDA. From the repository root:

```bash
make serve-transformers
```

In another terminal, use `make infer`. API docs are at
<http://127.0.0.1:8000/docs>. The client stays at `src/inference.py`, as required
by Assignment 7.

The new `src/server.py` reuses the same API with an Unsloth model loader.
`make serve` starts that implementation. Run only one server at a time on port
8000; `/health` identifies the backend as `transformers-peft` or `unsloth`.
