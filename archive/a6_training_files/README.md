# A6 training files (all rounds)

JSONL packs used to train Llama-3.2-1B-Instruct LoRA for Assignment 6.
The exact stage order and CLI flags are documented in `src/trainer.py` (module docstring).

| Subfolder | Stage | Role |
|-----------|-------|------|
| `01_public_base/` | optional early | public / base splits (superseded by hybrid for the final chain) |
| `02_hybrid/` | **1** | fresh LoRA — `hybrid_train.jsonl` / `hybrid_dev.jsonl` |
| `03_continue/` | **2** | continue from hybrid best |
| `04_focus/` | **3** | continue from continue best |
| `05_focus2/` | **4** | continue from focus best → **final adapter for answers.ipynb** |
| `evals_snapshots/` | — | graded-10 dumps after some runs |
| `gold_answers.json` | — | local keyword gold |

Shared LoRA knobs every stage: `r=4`, `alpha=8`, `q_proj,v_proj`, `lr=2e-4`, `seed=0`.

| Stage | batch × accum | epochs | grad checkpoint | init-adapter |
|-------|---------------|--------|-----------------|--------------|
| 1 hybrid | 1 × 8 | 3 | on (default) | none |
| 2 continue | 512 × 1 | 3 | **off** | stage1 best |
| 3 focus | 512 × 1 | 6 | **off** | stage2 best |
| 4 focus2 | 256 × 1 | 8 | **off** | stage3 best |
