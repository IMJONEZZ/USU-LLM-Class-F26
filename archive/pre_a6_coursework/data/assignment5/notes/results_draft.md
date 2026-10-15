# Assignment 5 — results and reflection draft for review

Status as of October 7, 2026: implementation and CPU validation completed.
**GPU integration, GPU feasibility, Llama baseline, full training and final Llama
evaluation are unverified.** No task-performance improvement is claimed.
No PR comment or initial Action Plan was modified.

## Completed checks

The Python checks below were run in host environments unless identified otherwise.
The Docker migration adds `Dockerfile.training` and a build-context allowlist,
while keeping the inference Dockerfile, application code and CPU CI intact.

- Verified the published Linux/amd64 Python 3.12.14 and uv 0.12.10 image manifests;
  their digests are pinned in the training Dockerfile.
- Checked Bash syntax for all 13 command blocks in the Docker guide.
- Re-ran Ruff and the full project suite after the Docker changes: the results
  below still pass. Also re-ran the 35 shared CPU tests from `/tmp` with an explicit
  source path, matching the image's working-directory arrangement; 1 GPU test skipped.
- The user successfully built the first training image with `sudo docker build`:
  all 16 build stages completed, including dependency installation and `uv pip check`.
- The first container CPU test run failed during collection: the host UID 1000
  had no passwd entry inside the image, and TorchInductor's username lookup raised
  `KeyError`. A second import then failed with an already-registered artifact.
  These were two collection errors, not completed tests.
- The corrected image supplies a login name for Python's lookup and writable
  `/tmp/a5-cache` defaults; experiment mounts use that path. Host bytecode is also
  explicitly excluded from the build context. A fresh-process missing-passwd-entry
  simulation reproduced the original error; with the fix, real LoRA backward and
  all 35 shared CPU tests passed, with 1 GPU test skipped.
- The user's subsequent container run with the host UID/GID passed: **35 passed,
  1 skipped, 3 warnings in 10.25 seconds**. The warnings concern missing pretrained
  config directories for locally initialized test models. The skipped test is
  opt-in GPU integration. This validates the shared CPU workflow in the corrected
  image; container GPU access, mounted preparation and GPU feasibility remain pending.
- The assistant still cannot access the Docker socket, and noninteractive sudo
  requires authentication, so container commands are being validated by the user.
- `training/.venv` remains available until container validation succeeds. No
  existing corpus, manifest, reports, caches or checkpoints were removed.

- `uv run ruff check .`: passed.
- `uv run ruff format --check .`: passed.
- `uv run pytest`: 347 passed, 3 skipped; overall coverage 94.31% (80% required).
  Skips are CUDA-dependent tests, including the opt-in Unsloth integration test.
- Installed the separate Python 3.12 training environment from its resolved lock.
  `uv pip check --python training/.venv/bin/python`: passed, 117 packages compatible.
- Shared training, Llama evaluation and report tests in that environment:
  35 passed, 1 GPU integration test skipped. CPU tests use a locally initialized
  tiny Llama with real PEFT adapters and no model/data downloads.
- Actual tests verify answer-only loss, token-weighted accumulation, finite loss,
  adapter changes, unchanged base weights, best-development checkpoint selection,
  checkpoint reload logits and greedy generation. These are correctness tests,
  **not Star Wars model-performance experiments**.
- Real-corpus preprocessing ran with the cached Llama/BERT tokenizers, using the
  separate environment and `HF_HUB_OFFLINE=1`. No Llama or BERT model weights were
  loaded for it. The Llama revision resolved to
  `4e20de362430cd3b72f300e6b0f18e50e7166e08`.

## Data audit

| Partition | Original records | Duplicate exclusions | Retained records | Eligible reconstruction examples | Insufficient words |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 2,018 | 48 | 1,970 | 976 | 994 |
| Dev | 252 | 2 | 250 | 108 | 142 |
| Test | 253 | 0 | 253 | 133 | 120 |

No retained record was excluded for sequence length with the default limits.
The longest complete Llama training-format sequence across all partitions was
190 tokens. These counts measure preparation, not model accuracy.

Matching uses exact raw dialogue (`Line`) strings, ignoring speakers and retaining
case/punctuation/whitespace. The original manifest stays authoritative and unchanged.
Test takes priority over dev, and dev/test take priority over training. The audit
found 25 distinct raw dialogue strings crossing partitions; only one had at least
eight lexical words: “Shut down all the garbage mashers on the detention level.”
Its training record was excluded and its development record retained.

[The checked-in audit](data_audit.json) lists every duplicate-excluded ID, its reason,
conflicting partitions, corpus/manifest hashes and resulting counts. The full local
preprocessing artifact, including every ineligible example ID/reason and fixed
span, is `data/assignment5/cpu-prepared/prepared.json`. Host experiment commands
create their own immutable artifact in `data/assignment5/prepared`.

## Question 2 measurements still pending

| Measurement | Before | After |
| --- | --- | --- |
| Exact-span match | Not run | Not run |
| ROUGE-1 / ROUGE-L | Not run | Not run |
| Smoothed bigram BLEU | Not run | Not run |
| BERTScore precision / recall / F1 | Not run | Not run |
| Development loss | Not run | Not run |

Training time, peak allocated/reserved VRAM, training-loss changes and selected
checkpoint are not measured. Representative Llama successes/failures are unavailable
until evaluation runs. `nvidia-smi` could not communicate with the driver in the
assistant session, and PyTorch reported CUDA unavailable. The host GPU recently
worked according to the user; no attempt was made to reinstall or change drivers.
The failed preflight and configuration are preserved in
`data/assignment5/session-gpu-check/failure.json`; it stopped before importing
Unsloth or loading model weights.

The user subsequently confirmed host GPU visibility: Quadro T2000, compute
capability 7.5, CUDA 12.8 and about 3.50 GiB free. This establishes host visibility,
not successful container training.

Run the [Docker commands](README.md#run-the-experiment-in-docker) to produce real
measurements, then use `src.training_report` to generate
`data/assignment5/docker-run/results.md` through the persistent output mount.
That report compares identical prepared examples/settings, includes loss history,
time/VRAM, and selects the first two successes/failures in fixed example order.
Those examples illustrate outputs; they are not a statistical sample.

Loss measures answer-token/EOS cross-entropy under the reference context. Lower
loss means higher probability assigned to those tokens. Lower training loss alone
can reflect memorization and does not demonstrate better held-out generation.
The dev split selects checkpoints, while the test split measures the final result.
Do not compare old BERT token accuracy directly with Llama text metrics.

## Other datasets to consider

These were researched, not used for training:

- [DailyDialog](https://aclanthology.org/I17-1099/) provides multi-turn everyday
  conversations. Applying the same four-word masking task could broaden phrasing
  beyond Star Wars dialogue, but its everyday domain differs from film scripts.
- [Cornell Movie-Dialogs](https://convokit.cornell.edu/documentation/movie.html)
  provides movie dialogue and conversation/movie metadata. It could contribute
  more script-style examples, but overlapping Star Wars films or quotations must
  be excluded to protect this evaluation.

Those benefits are hypotheses. Preserve conversation/movie boundaries and audit
text overlap with the established dev/test partitions before using either corpus.

## Implementation reflection notes for review

The implementation follows the selected base model, ordinary FP16 LoRA, fixed spans,
answer-only loss and development-only selection. It reuses the previous evaluator's
split validation/span selection/text metrics and the data loader's seeded batching.
It adds a collator because the old custom-BPE, already-shifted pairs are unsuitable
for prompt reconstruction with Llama's tokenizer.

Environment validation required two compatibility fixes: deferred annotations in
`bpe_tokenizer.py` for Python 3.12, and a TorchAO pin compatible with PyTorch 2.9.
CPU CI keeps the project's original Python/runtime stack. The loop uses PyTorch
AdamW directly; TRL is present as an Unsloth dependency, but its SFTTrainer is not
needed. Small adapter weights use FP32 for stable scaled optimization; the frozen
base remains FP16. Standard checkpointing avoids automatic CPU offloading.

The tests target leakage, label boundaries, optimizer behavior, preservation of
base weights, checkpoint reload and reproducible evaluation because a script can
execute successfully while getting these details wrong. Existing BERT behavior
continues to pass its tests. GPU kernels and memory feasibility still require the
explicit host integration run; package installation and CPU tests cannot establish
that a 4 GB GPU can train this configuration.

Further work after the agreed experiment could include repeated seeds and
uncertainty estimates, analysis by span difficulty, and carefully deduplicated
additional dialogue data. These are suggestions for later work, not completed
experiments or additions to the student's initial Action Plan.
