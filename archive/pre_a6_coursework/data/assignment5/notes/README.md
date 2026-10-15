# Assignment 5: four-word reconstruction with Llama

Use Docker for Assignment 5, continuing the Assignment 4 workflow. The dedicated
[`Dockerfile.training`](../Dockerfile.training) installs Python 3.12, Unsloth and
all pinned training dependencies **inside the image**. Host Python and
`training/.venv` are not needed to run it. The original `Dockerfile` remains the
Python 3.14 Assignment 4 inference image, and CPU CI keeps its existing setup.

The experiment remains **meta-llama/Llama-3.2-1B (base), before versus after ordinary
FP16 LoRA**. The host GPU preflight succeeded according to the user's output, but
training feasibility and task metrics remain unverified. No drivers were changed.
See [the results draft](results_draft.md) for the exact validation status.

## Build and validate the training image

Run these commands from the repository root in a host terminal that can access
Docker. If your Assignment 4 workflow requires `sudo docker`, use that same prefix
for the commands below. Do not change socket permissions or reinstall drivers.
Docker GPU commands require the NVIDIA Container Toolkit already used in Assignment 4.
When forwarding `HF_TOKEN` through sudo, use `sudo --preserve-env=HF_TOKEN docker run`;
the read-only token-file mount option works with ordinary `sudo docker run`.

```bash
docker build --platform linux/amd64 -f Dockerfile.training -t usu-llm-training:a5 .
docker run --rm --entrypoint uv usu-llm-training:a5 pip check --system
docker run --rm --user "$(id -u):$(id -g)" usu-llm-training:a5 \
  -m pytest -c /app/pyproject.toml --no-cov -p no:cacheprovider -q \
  /app/tests/test_trainer.py /app/tests/test_llama_evaluator.py \
  /app/tests/test_training_report.py /app/tests/test_trainer_gpu.py
```

The CPU test command requires no GPU, credentials, dataset mounts or model downloads.
The GPU integration test is skipped there. This checks the shared training loop,
real tiny-model LoRA updates, adapter reload and evaluation inside the actual image.
`--no-cov` applies only to this subset; full CPU CI still enforces 80% coverage.
The default image command is `python -m src.trainer --help`; arguments to
`docker run` follow the image's Python entrypoint.
The image supplies a stable login name for Python libraries when the host UID has
no `/etc/passwd` entry. Unmounted validation runs use writable caches under
`/tmp/a5-cache`; experiment runs bind-mount that same path for persistence.

The dependency lock targets Linux x86-64 and pins Unsloth/Unsloth Zoo 2026.10.1,
PyTorch 2.9.1 (CUDA 12.8 wheels), Transformers 4.57.6, PEFT 0.18.0, TRL 0.22.2,
xFormers 0.0.33.post2 and TorchAO 0.14.1. The image pins Python 3.12.14 and uv
0.12.10. CUDA userspace libraries come from the lock; NVIDIA drivers stay on the
host. A compiler is installed for Triton. The Python 3.14 project lock and inference
Dockerfile are unchanged by this migration.

The build context uses the allowlist in `Dockerfile.training.dockerignore`.
Only application/test Python files, project test configuration and the training
lock enter the image. Datasets, model weights, reports, `.venv`, caches and
credentials are excluded. Authentication is used only at container runtime.

Sources: [uv's Docker guide](https://docs.astral.sh/uv/guides/integration/docker/),
[Docker GPU access](https://docs.docker.com/engine/containers/resource_constraints/#gpu),
[Unsloth installation](https://unsloth.ai/docs/get-started/install/pip-install),
[Unsloth requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements),
[official Llama 3.2 notebook](https://github.com/unslothai/notebooks/blob/main/nb/Llama3.2_%281B_and_3B%29-Conversational.ipynb)
and [TorchAO compatibility](https://github.com/pytorch/ao/issues/2919).
The notebook's Instruct model, conversational format and quantization are not used.
TRL is a dependency; the assignment uses the shared PyTorch loop rather than SFTTrainer.

## Mount data, outputs, cache and authentication

Use Bash for the following arrays, and keep the same shell for subsequent commands.
The output default is a **new** `data/assignment5/docker-run` directory so earlier
preparation, reports and checkpoints remain untouched. Choose another fresh directory
for a new experiment; commands refuse to overwrite existing outputs.

```bash
A5_CORPUS="$PWD/SW_EpisodeIV_VI.json"
A5_SPLITS_DIR="$PWD/data/evaluation"
A5_OUTPUT_DIR="$PWD/data/assignment5/docker-run"
A5_CACHE_DIR="$PWD/.cache/assignment5-docker"

test -f "$A5_CORPUS"
test -f "$A5_SPLITS_DIR/splits.json"
mkdir -p "$A5_OUTPUT_DIR" "$A5_CACHE_DIR"

A5_DOCKER_OPTS=(
  --rm
  --user "$(id -u):$(id -g)"
  --mount "type=bind,src=$A5_CORPUS,dst=/inputs/corpus.json,readonly"
  --mount "type=bind,src=$A5_SPLITS_DIR,dst=/inputs/evaluation,readonly"
  --mount "type=bind,src=$A5_OUTPUT_DIR,dst=/outputs"
  --mount "type=bind,src=$A5_CACHE_DIR,dst=/tmp/a5-cache"
)
# Forward an already-set token without putting its value in the command line.
A5_AUTH=(-e HF_TOKEN)
```

Stop if either `test -f` fails and set the correct existing paths. Docker's
`--mount` also rejects missing bind sources instead of creating empty inputs.
The corpus and split directory are read-only. Reports/checkpoints are written to
`/outputs`, which persists at `$A5_OUTPUT_DIR`; the container runs as your host
UID/GID so those files stay yours. Model and compilation caches persist separately
under `$A5_CACHE_DIR`. Do not mount the whole repository over `/app`, which would
hide the image's source files.

If you use an existing Hugging Face CLI login instead of `HF_TOKEN`, replace the
`A5_AUTH` array with a read-only mount of its token file:

```bash
A5_TOKEN_FILE="${HF_TOKEN_PATH:-${HF_HOME:-$HOME/.cache/huggingface}/token}"
test -f "$A5_TOKEN_FILE"
A5_AUTH=(
  --mount "type=bind,src=$A5_TOKEN_FILE,dst=/run/secrets/hf_token,readonly"
  -e HF_TOKEN_PATH=/run/secrets/hf_token
)
```

Stop if the token file is absent and use your existing authentication setup.
No token values are printed or copied into the build context/image. The separate
container model cache does not require copying a credential into it.
[Hugging Face documents `HF_TOKEN` and `HF_TOKEN_PATH` here](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables).

## Run the experiment in Docker

1. Check that Docker can access the GPU. This verifies runtime visibility;
   the feasibility command still needs to execute actual optimizer steps.

   ```bash
   docker run --gpus all "${A5_DOCKER_OPTS[@]}" usu-llm-training:a5 \
     -c 'import json; from src.trainer import gpu_environment; print(json.dumps(gpu_environment(), indent=2))'
   ```

2. Prepare the real corpus with the existing manifest. No model weights are needed.

   ```bash
   docker run --gpus all "${A5_DOCKER_OPTS[@]}" "${A5_AUTH[@]}" usu-llm-training:a5 \
     -m src.trainer prepare \
     --corpus /inputs/corpus.json \
     --splits /inputs/evaluation/splits.json \
     --output /outputs/prepared
   ```

   The resolved immutable Llama revision, tokenizers, exact spans and labels are
   saved. To reproduce the earlier CPU preprocessing model revision, add
   `--revision 4e20de362430cd3b72f300e6b0f18e50e7166e08`.
   The BERT span tokenizer is pinned to the Assignment 3 revision.

3. Run feasibility: two real optimizer updates over the longest training examples,
   development loss, and an adapter save. Success requires adapter weights to change.

   ```bash
   docker run --gpus all "${A5_DOCKER_OPTS[@]}" "${A5_AUTH[@]}" usu-llm-training:a5 \
     -m src.trainer feasibility \
     --prepared /outputs/prepared \
     --output /outputs/feasibility
   ```

4. Evaluate the untouched base model on all eligible test examples.

   ```bash
   docker run --gpus all "${A5_DOCKER_OPTS[@]}" "${A5_AUTH[@]}" usu-llm-training:a5 \
     -m src.evaluator llama \
     --prepared /outputs/prepared \
     --output /outputs/before.json
   ```

5. Train for up to three epochs and select the lowest development loss. A matching
   successful feasibility run is required. Its adapter is discarded; full training
   starts from the original base.

   ```bash
   docker run --gpus all "${A5_DOCKER_OPTS[@]}" "${A5_AUTH[@]}" usu-llm-training:a5 \
     -m src.trainer train \
     --prepared /outputs/prepared \
     --feasibility /outputs/feasibility \
     --epochs 3 \
     --output /outputs/train
   ```

6. Evaluate the selected adapter with identical inputs and greedy decoding.

   ```bash
   docker run --gpus all "${A5_DOCKER_OPTS[@]}" "${A5_AUTH[@]}" usu-llm-training:a5 \
     -m src.evaluator llama \
     --prepared /outputs/prepared \
     --adapter /outputs/train/best \
     --output /outputs/after.json
   ```

7. Generate the results draft. This step needs no GPU or authentication.

   ```bash
   docker run "${A5_DOCKER_OPTS[@]}" usu-llm-training:a5 \
     -m src.training_report \
     --before /outputs/before.json \
     --after /outputs/after.json \
     --training /outputs/train/training.json \
     --output /outputs/results.md
   ```

   The report is now at `data/assignment5/docker-run/results.md` on the host.
   Best adapter files are under `data/assignment5/docker-run/train/best`.

To run the opt-in real Unsloth pytest after preparation:

```bash
docker run --gpus all "${A5_DOCKER_OPTS[@]}" "${A5_AUTH[@]}" \
  -e RUN_A5_GPU=1 -e A5_PREPARED=/outputs/prepared usu-llm-training:a5 \
  -m pytest -c /app/pyproject.toml --no-cov -p no:cacheprovider -q \
  --basetemp=/outputs/gpu-test-tmp /app/tests/test_trainer_gpu.py
```

Use a fresh `--basetemp` for each GPU test: pytest clears that directory on reuse.
The ordinary feasibility CLI is preferable for persistently named experiment results.

CPU CI still runs the unchanged commands outside the training image:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Which training files stay, and when to remove the old environment

- `requirements.lock`: required by the Docker build; exact dependency versions.
- `requirements.in`: maintained source for intentionally updating the lock.
- `README.md`: Docker commands and explanations.
- `data_audit.json` and `results_draft.md`: reviewable assignment evidence;
  they are not runtime dependencies and do not enter the image.
- `training/.venv`: the old ignored host environment. It is not used by Docker.
  It has been **retained** until the full container workflow is validated.
  The corrected image now passes the container CPU tests with the host UID/GID;
  mounted preparation and GPU feasibility are still pending. Remove only
  this environment after the image builds, container CPU tests pass, preparation
  writes valid outputs through the mounts, and GPU feasibility succeeds.
  Do not remove `data/`, reports, checkpoints or the `training/` directory itself.

Until container validation succeeds, the existing ignored environment is available
as a rollback; this does not require creating a new host environment. No local
environment or artifact has been deleted during this migration.

The lock can be deliberately regenerated inside the built image, without host uv:

```bash
docker run --rm --user "$(id -u):$(id -g)" --entrypoint uv \
  --mount "type=bind,src=$PWD/training,dst=/requirements" usu-llm-training:a5 \
  pip compile /requirements/requirements.in --python-version 3.12 \
  --python-platform x86_64-unknown-linux-gnu --output-file /requirements/requirements.lock
```

Rebuild and revalidate after a lock update. Normal experiments use the committed
lock as-is. Unsloth GPU integration and VRAM feasibility remain separate from a
successful dependency installation or CPU test run.

## Data and token boundaries

The manifest is required and never regenerated or reassigned. Duplicate detection
compares raw `Line` strings exactly, ignoring speakers but preserving case,
punctuation, Unicode representation and whitespace. Test records stay unchanged;
dev records matching original test text are excluded; train records matching
original dev or test text are excluded. Within-partition duplicates remain.
The filter runs before any span preparation, baseline inference or training.
All subsequent commands use the same checksummed preparation artifact.

`prepared.json` records original and retained record counts, duplicate-excluded
IDs/reasons/conflicting partitions, eligible example counts, and every
span/length exclusion with its reason. An integrity hash detects changes to the
artifact. The original dataset and manifest hashes are recorded. Repeating the
experiment after changing data/settings requires new preparation and new reports.

Assignment 3's span selector is reused, including its 256-BERT-token eligibility
limit, four lexical words, two visible words on either side, contractions,
hyphenation, punctuation and deterministic identity/seed selection. Only its
pinned tokenizer is used: no BERT model is loaded for preprocessing. Reference
text is sliced from the original character offsets to retain case and punctuation.
Llama never receives custom BPE or BERT token IDs.

The new task does not fit the original `TextDataset`'s already-shifted next-token
pairs. The loader's seeded PyTorch batching policy is shared through
`seeded_loader`; `collate_answers` adds padding, attention masks and answer-only
labels for the new examples. The model handles its own causal label shift.

The prompt is:

```text
Recover the four missing words. Return only the missing text.

Dialogue:
{prefix}[MISSING SPAN]{suffix}

Missing text:
```

Prompt and answer are tokenized separately, then concatenated, with a final EOS.
This deliberate boundary prevents a cross-boundary BPE merge from changing the
last prompt token between training and generation. Only the prompt uses tokenizer
special-token insertion (BOS). Labels are `-100` for prompt/padding and target
IDs for the answer/EOS. EOS remains supervised even when it is also the pad ID.
Overlength examples are reported, never truncated. A fixed 32-token generation
budget is reserved during preparation; generation never consults answer length.

## Training and measurements

Defaults are batch size 1, accumulation 8, rank 4, alpha 8, q_proj/v_proj adapters,
zero adapter dropout, AdamW learning rate 0.0002, sequence length 256 and three
epochs. Override `--batch-size`, `--accumulation`, `--learning-rate`, `--rank`,
`--alpha`, `--targets` and `--epochs` as needed. Repeat feasibility with the same
options before a full run. Smaller adapters and short sequences reduce memory;
**fitting a full FP16 1B base and training on 4 GB is not guaranteed**.

The frozen base is FP16. Only adapter parameters require gradients; small adapter
weights and optimizer state use FP32 with CUDA FP16 autocast and loss scaling.
Standard gradient checkpointing is enabled; automatic embedding offloading,
Unsloth's offloading checkpoint mode, 4/8-bit loading and full fine-tuning are off.
Model name, revision, placement, dtype and trainable names are checked. No method
changes or retry with quantization/offloading occur after an OOM.

Training/dev losses are answer-token-weighted, including EOS. Accumulation weights
each microbatch by its supervised token count; the final partial group is retained.
Development loss is measured initially and after each epoch; the minimum
post-training value wins, with an earlier checkpoint winning ties. Test metrics
never enter checkpoint selection. The saved adapter contains the exact base
revision metadata and a tokenizer copy; use it with the matching preparation.

`training.json` contains parameter counts/names, epoch training/dev losses,
optimizer step counts, chosen epoch, package versions, configuration, GPU metadata,
training wall time and peak allocated/reserved bytes. CUDA synchronization and
peak resets bracket training loops. Cached allocations from development are
released before resetting training peaks. Time includes batching/backward/optimizer
work; it excludes model loading, development/test inference and checkpoint saving.
The baseline loss is initial development loss; epoch training losses average a
model that is changing during the epoch, not repeated measurements of one state.

## Evaluation and failure handling

The original BERT CLI remains available unchanged. `src.evaluator llama` is the
separate text-generation path. Both baseline and final evaluation use standard
Transformers FP16 loading, left padding, explicit attention masks, one beam and
greedy decoding. The evaluator decodes only generated tokens. It strips the
protocol EOS and padding following EOS; other generated special tokens remain.
Raw decoded text, token IDs, normalized text, empty/length-limit status and
format issues (unexpected word count or generated special tokens) are saved.
Extra words/explanations are never removed or clipped to four words.

Exact match strips/collapses whitespace but retains case and punctuation.
ROUGE-1/L use mean F1 without stemming; corpus BLEU uses smoothed bigrams;
BERTScore uses roberta-base, no IDF and no baseline rescaling, matching the previous
settings. Empty outputs remain in the denominator with zero ROUGE/BERTScore;
they stay in corpus BLEU, with an explicitly defined zero for an all-empty corpus.
Llama is released before metric models load, and BERTScore always runs on CPU.
Raw predictions are saved before scoring, so metric-download failure preserves them.
No BERT token accuracy or new word accuracy is reported for Llama.

On local OOM or another training error, inspect the sanitized `failure.json` and
`config.json` if loading reached that stage. Do not edit drivers or change precision
as a workaround. The next Modal step is to use a larger NVIDIA GPU, use the
same training Docker image, transfer the real corpus/manifest/prepared artifact, use
Modal's secret mechanism for existing Hugging Face access, then rerun feasibility
and the same baseline/train/final commands in fresh output locations. No paid
cloud work, deployment, or credential transfer is launched by this implementation.

All large runtime artifacts are under ignored `data/`; environment/cache directories
are ignored too. This work does not modify the original PR Action Plan, post
comments, commit, or push. Review results and reflection text before posting it.
