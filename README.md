# USU LLM Class — Fall 2026

Homework-submission repository for USU LLM Class DSAI-5810/6810.

> Assignments and course materials will be added throughout the Fall 2026 semester.

## Contributing

### Set up your environment

Requires Python 3.14 or newer.

Install `uv` using the instructions found here: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

Then sync the project:

```bash
uv sync
```

### Start experimenting

Run the entrypoint script:

```bash
uv run src/main.py
```

If you prefer a Jupyter notebook:

```bash
uv run --with jupyter jupyter lab
```

### Set up pre-commit (optional)

CI runs the project checks for every push and pull request. To run those checks before committing, install the hooks:

```bash
uvx prek install
```

### Run checks locally

Tests use small fixtures and mocked services without model downloads. The GPU
inference tests check script behavior with mocks and run a tiny, locally created
model on CUDA when available; that test is skipped on CPU-only machines. ZenML
tests exercise the real Iris training and evaluation steps and prepare the
pipeline graph without submitting a run or requiring a server or store.

Pytest measures coverage of `src` and requires at least 80% overall coverage.

```bash
uv run ruff check
uv run ruff format --check .
uv run pytest
```

### AI assistance

The course supports Claude Code and Codex. Both assistants follow the same student-learning and academic-integrity expectations in [AGENTS.md](AGENTS.md).

## Progress

- A0 - Basic Tokenizer (done)
- A1 - BPE Tokenizer (done)
- A2 - Data Loader (done)
- A3 - Evaluators (done)
- A4 - MLOps (done)
- A5 - Training a Model (In Progress)

## Additional Usage

### Tokenization

Train a byte-level BPE tokenizer on a dialogue corpus. Replace `CORPUS` with your
JSON dataset:

```bash
uv run python -m src.bpe_tokenizer CORPUS --vocab-size 4096
```

The dataset must be a JSON array with string `Character` and `Line` fields:

```json
[{"Character": "NARRATOR", "Line": "The trees are green."}]
```

`load_corpus` joins both fields from every record with newlines, preserving
character names and dialogue. Convert other schemas to these field names before
loading. In Python, `load_corpus(path)` requires an explicit dataset location;
a missing dataset raises `FileNotFoundError`.

The vocabulary size defaults to 4,096, including the 256 initial byte tokens.
Small corpora may finish below the requested size.

To save and reuse a trained tokenizer, replace `TOKENIZER` with your chosen
output filename:

```bash
uv run python -m src.bpe_tokenizer CORPUS --vocab-size 4096 --save TOKENIZER
uv run python -m src.bpe_tokenizer --load TOKENIZER --text "Hello, café🙂!"
```

Saving is optional, and loading a saved tokenizer does not require the original
training dataset. Saved tokenizers contain hexadecimal token bytes and ordered
merge rules in JSON. In Python, use `tokenizer.save(path)` and
`BPETokenizer.load(path)`. `--load` cannot be combined with a corpus argument or
`--vocab-size`. See `uv run python -m src.bpe_tokenizer --help` for all options.

### Data Loading

After `uv sync`, train a tokenizer and prepare next-token batches in a Python
session. Replace `CORPUS` with your JSON dataset:

```python
from src.bpe_tokenizer import BPETokenizer, load_corpus
from src.dataloader import create_dataloader

text = load_corpus("CORPUS")
tokenizer = BPETokenizer()
tokenizer.train(text, vocab_size=4096)
loader = create_dataloader(text, tokenizer, max_length=256, batch_size=8, seed=0)
inputs, targets = next(iter(loader))
print(inputs.shape, targets.shape)  # Each full batch has shape [8, 256].
```

`TextDataset` encodes the text once, preserving character names, dialogue, and
newlines. Each example uses 257 source tokens to produce 256 input tokens and
256 targets shifted forward by one token. Set `max_length` to change this length.
Text with fewer than `max_length + 1` encoded tokens raises an error; use more data
or a smaller `max_length`. The one-record JSON above illustrates the schema and
is too small for this batching example.

`stride` defaults to `max_length` and must satisfy
`1 <= stride <= max_length`. Smaller strides create overlapping windows.
When needed, one additional full window includes the final corpus token as a
target, avoiding padding or dropping the tail. This final window may overlap.

The loader shuffles examples by default, preserves token order within each
example, and keeps the final smaller batch. Recreating the loader with the same
seed (default `0`) reproduces its shuffle sequence in the same environment;
reusing it for another epoch advances that sequence. Use `shuffle=False` to
inspect examples in corpus order.

Batches contain CPU `torch.long` tensors with shape `[batch_size, max_length]`,
except that the final batch may have fewer examples. A training loop can move
batches to a GPU when a compatible PyTorch build and driver are available.

Individual byte-level windows can split a Unicode character, so they may not
decode independently. A complete source window consists of one input sequence
plus its last target token.

### BERT Evaluation

The evaluator uses `google-bert/bert-base-uncased` to reconstruct four consecutive
masked words with at least two visible words on either side. It uses BERT's
WordPiece tokenizer and runs without fine-tuning. Only dialogue text enters the
model. Reconstruction uses the original span's WordPiece count.

Evaluate a small development sample or the full eligible test split:

```bash
uv run python -m src.evaluator CORPUS --split dev --sample-size 8
uv run python -m src.evaluator CORPUS --split test
```

The corpus uses the JSON schema above. The first run downloads the required
model checkpoints and metric implementations. CPU is the default; use
`--device cuda` for a compatible GPU or `--strict-only` to skip text similarity
metrics. See `uv run python -m src.evaluator --help` for all options.

Evaluation uses persistent train/development/test partitions of approximately
80/10/10, grouping by conversation or scene when available. A fixed seed controls
mask selection. Reports include predictions, scores, exclusions, and settings
needed to reproduce the run.

#### Results

The recorded Star Wars dialogue evaluation on September 23, 2026 used 133 eligible
test examples with 698 masked token positions. Another 120 test lines were
excluded for containing fewer than eight words. The split contained 2,018 training
lines, 252 development lines, and 253 test lines. Training text supplied only the
frequency baseline; BERT was not fine-tuned.

Settings: seed 0, batch size 8, maximum input length 256 tokens.
BERT revision: `86b5e0934494bd15c9632b12f734a8a67f723594`.

| Metric | Result |
| --- | ---: |
| Token accuracy | 14.33% (100/698) |
| Exact-span match | 0.00% (0/133) |
| ROUGE-1 | 0.188563 |
| ROUGE-L | 0.188563 |
| BLEU | 0.087262 |
| BERTScore precision | 0.828137 |
| BERTScore recall | 0.815657 |
| BERTScore F1 | 0.821757 |

Text metrics compare reconstructed spans with their references. ROUGE reports
mean per-example F1 without stemming. BLEU uses up to bigrams with smoothing.
BERTScore uses `roberta-base` without IDF weighting or baseline rescaling; it
measures semantic similarity, not the percentage of correct reconstructions.

### GPU Inference with Docker

[src/gpu_inference.py](src/gpu_inference.py) runs causal language-model generation
on `cuda:0`. It loads weights in `float16`, transfers the entire model to the GPU,
and verifies that all parameters and buffers are on that device. Quantization
and CPU offloading are disabled, so the model must fit in GPU memory. CUDA is
required; the script raises an error when it is unavailable.

The [Dockerfile](Dockerfile) uses a Python 3.14 image with `uv`, installs build
tools for Triton, and installs locked runtime dependencies with
`uv sync --frozen --no-dev`. Its default command runs the inference script.

Build and run from the repository root on a host with an NVIDIA GPU, compatible
drivers, and Docker configured for GPU access through the NVIDIA Container
Toolkit:

```bash
docker build -t usu-llm-inference .
docker run --rm --gpus all \
  -e HF_TOKEN \
  -e PROMPT="Once upon a time" \
  -e MAX_NEW_TOKENS=50 \
  usu-llm-inference
```

For a model that requires authentication, set `HF_TOKEN` in your host environment
to a Hugging Face token with access to that model before running the container.
`-e HF_TOKEN` forwards that variable. Model files are downloaded on first use;
the image does not bundle model weights.

Configure the script using environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_ID` | `meta-llama/Llama-3.2-1B` | Hugging Face model ID or local model directory visible to the process. |
| `PROMPT` | `Once upon a time` | Text to continue. |
| `MAX_NEW_TOKENS` | `50` | Maximum number of generated tokens. |
| `TEMPERATURE` | `0.7` | Sampling temperature; generation uses sampling. |
| `PAUSE_SECONDS` | `0` | Pause after loading for inspection with `nvidia-smi`. |
| `HF_TOKEN` | Unset | Authentication token passed to the tokenizer and model loaders. |

Add `-e VARIABLE=value` before the image name to override a setting. To run
directly in the project environment with compatible CUDA support:

```bash
uv run python -m src.gpu_inference
```

The script prints environment details, model placement, generated text, and
allocated, reserved, and peak allocated GPU memory. A completed run prints
`RESULT: SUCCESS`. An exit handler reports the finish time and, when CUDA was
initialized, final memory statistics even if loading or generation failed.

### ZenML Iris Pipeline

[src/zenml_iris_pipeline.py](src/zenml_iris_pipeline.py) defines three connected
ZenML steps using scikit-learn's bundled Iris dataset:

1. `load_data` creates pandas features and labels, then splits the 150 examples
   into 120 training and 30 test examples with shuffling and `random_state=42`.
2. `train_model` fits an `SVC(gamma=0.001)` classifier on the training data.
3. `evaluate_model` computes test accuracy, prints it to four decimal places,
   and returns it as the `test_accuracy` output.

The project dependencies include pandas, scikit-learn, and `zenml[server]`.
After `uv sync`, initialize a ZenML repository once and run the pipeline from
the repository root:

```bash
uv run zenml init
uv run python -m src.zenml_iris_pipeline
```

The script submits `iris_training_pipeline` to the active ZenML stack. Local
repository configuration is stored in `.zen/`, which is ignored by Git. This
example uses Iris data and does not require the dialogue corpus or a GPU.

### Assignment 5: Docker LoRA training

`Dockerfile.training` packages Python 3.12 and the pinned dependencies in
`requirements-training.lock`. Maintain those pins through `requirements-training.in`.
The Assignment 4 inference Dockerfile and CPU CI remain unchanged. No host training
virtual environment is required.

Validated in Docker: **35 CPU tests passed, 1 opt-in GPU test skipped**, followed by
a successful real Unsloth feasibility run on the Quadro T2000: two optimizer steps
in 38.25 seconds of training-loop time, 2.46 GiB peak allocated / 2.69 GiB reserved,
and development loss 5.6116 → 5.5203. The adapter and success marker were saved.
The original-model baseline completed on 133 test examples: exact match 0/133,
ROUGE-1 0.04110, ROUGE-L 0.04052, BLEU 0.00785 and BERTScore F1 0.79750.
The saved report confirms no adapter was loaded; 124 outputs reached the fixed
generation limit. The first full training attempt failed before completing epoch 1
because strict gradient clipping interrupted FP16 loss-scale overflow recovery.
The loop now retries the same accumulation group at a reduced scale and preserves
the scaler across epochs; CPU regression tests cover recovery and persistent
failure. The corrected GPU run completed all three epochs (366 optimizer updates)
in 58 minutes 45 seconds of training-loop time, with one recovered overflow.
Development losses were 2.7907, 2.6454 and 2.5884, down from 5.6116 before training;
epoch 3 was selected. Peak GPU memory was 2.46 GiB allocated / 2.82 GiB reserved.
The measured run is saved under `data/assignment5/docker-run/train-retry1/`;
use its `best` adapter for final evaluation. Final evaluation completed on the same
133 test examples with matching model revision, prompts and generation settings:

| Metric | Original base | Selected adapter |
| --- | ---: | ---: |
| Exact-span match | 0/133 | 0/133 |
| ROUGE-1 | 0.04110 | 0.18425 |
| ROUGE-L | 0.04052 | 0.18425 |
| BLEU | 0.00785 | 0.10224 |
| BERTScore F1 | 0.79750 | 0.82560 |
| Four-word outputs | 0/133 | 122/133 |
| Outputs reaching the generation limit | 124/133 | 0/133 |

The adapter improved output format and partial text similarity, but did not recover
any complete reference span exactly. BERTScore is not a percentage accuracy, and
lower development loss does not establish exact reconstruction success. The
measured comparison is saved in `data/assignment5/docker-run/results.md`.
Local reports and the detailed notes are
preserved under ignored `data/assignment5/`; they are not files to push.

Build when the source or dependencies change. Rebuild to include the overflow fix:

```bash
sudo docker build --platform linux/amd64 -f Dockerfile.training -t usu-llm-training:a5 .
sudo docker run --rm --user "$(id -u):$(id -g)" usu-llm-training:a5 \
  -m pytest -c /app/pyproject.toml --no-cov -p no:cacheprovider -q \
  /app/tests/test_trainer.py /app/tests/test_llama_evaluator.py \
  /app/tests/test_training_report.py /app/tests/test_trainer_gpu.py
```

From the repository root, paste this into Bash once. It defines `a5` to run Python
inside Docker with read-only inputs and persistent outputs/cache owned by your UID.
Choose a fresh output directory for each experiment; existing outputs are never
silently overwritten. Authentication uses your existing `HF_TOKEN` or login token
file at runtime; no credentials enter the image.

```bash
A5_OUTPUT_DIR="$PWD/data/assignment5/docker-run"
A5_CACHE_DIR="$PWD/.cache/assignment5-docker"
mkdir -p "$A5_OUTPUT_DIR" "$A5_CACHE_DIR"
A5_AUTH=()
A5_TOKEN_FILE="${HF_TOKEN_PATH:-${HF_HOME:-$HOME/.cache/huggingface}/token}"
if [[ -f "$A5_TOKEN_FILE" ]]; then
  A5_AUTH=(--mount "type=bind,src=$A5_TOKEN_FILE,dst=/run/secrets/hf_token,readonly"
           -e HF_TOKEN_PATH=/run/secrets/hf_token)
fi

a5() {
  sudo --preserve-env=HF_TOKEN docker run --rm --gpus all \
    --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$PWD/SW_EpisodeIV_VI.json,dst=/inputs/corpus.json,readonly" \
    --mount "type=bind,src=$PWD/data/evaluation,dst=/inputs/evaluation,readonly" \
    --mount "type=bind,src=$A5_OUTPUT_DIR,dst=/outputs" \
    --mount "type=bind,src=$A5_CACHE_DIR,dst=/tmp/a5-cache" \
    -e HF_TOKEN "${A5_AUTH[@]}" usu-llm-training:a5 "$@"
}
```

Run these **one at a time**, proceeding only after the previous command succeeds:

```bash
# Prepare fixed spans using the real data and existing split manifest.
a5 -m src.trainer prepare --corpus /inputs/corpus.json \
  --splits /inputs/evaluation/splits.json --output /outputs/prepared

# Two real optimizer updates; a success marker is required before full training.
a5 -m src.trainer feasibility --prepared /outputs/prepared --output /outputs/feasibility

# Evaluate the unmodified base model.
a5 -m src.evaluator llama --prepared /outputs/prepared --output /outputs/before.json

# Train from the original base; select the lowest development-loss checkpoint.
a5 -m src.trainer train --prepared /outputs/prepared \
  --feasibility /outputs/feasibility --epochs 3 --output /outputs/train

# Evaluate the selected adapter, then produce a measured results draft.
a5 -m src.evaluator llama --prepared /outputs/prepared \
  --adapter /outputs/train/best --output /outputs/after.json
a5 -m src.training_report --before /outputs/before.json --after /outputs/after.json \
  --training /outputs/train/training.json --output /outputs/results.md
```

Results/checkpoints persist at `$A5_OUTPUT_DIR`, including `train/best`,
`training.json` inside `train`, and `results.md`. Failures retain configuration and
sanitized errors. An OOM never triggers quantization, offloading or paid cloud work.
For a retry after failure, keep the failed directory and choose a fresh output such
as `/outputs/train-retry1`; use that same path for subsequent evaluation/reporting.
When piping commands to `tee`, enable `set -o pipefail` so a failed container still
returns a nonzero exit status.
If it cannot fit locally, use the same image and data on a larger GPU after deciding
whether to incur that cost. Keep the same preparation and settings for comparisons.

The task uses **meta-llama/Llama-3.2-1B base**, frozen FP16 weights, rank-4 LoRA on
q_proj/v_proj, batch size 1, accumulation 8, sequence limit 256 and at most 3 epochs.
Only answer tokens and EOS contribute to loss; the model performs the causal shift.
FP16 training uses a persistent dynamic loss scaler. On gradient overflow, no
optimizer update occurs; the whole accumulation group is retried at a lower scale
(at most 32 reductions before failure). Epoch reports count successful optimizer
steps and overflow retries separately. Non-finite losses still fail immediately.
This follows [PyTorch's AMP overflow handling](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
while replaying skipped batches to retain all training examples.
Use the CLI `--help` for configurable settings. Checkpoint selection uses dev loss,
never test metrics. BERT evaluation remains available through its original CLI.

Exact raw dialogue matches are excluded with test → dev → train priority, without
changing the manifest. The preparation artifact records all excluded IDs/reasons,
counts, fixed character spans, tokenizers and the immutable model revision.
Llama uses only its own token IDs. Greedy evaluation reuses those prompts and a
fixed generation budget; exact match collapses whitespace but retains case and
punctuation. ROUGE, BLEU and BERTScore keep the previous settings; empty outputs
remain in accounting, and BERTScore runs on CPU after releasing Llama.

Keep source, tests, Dockerfiles and dependency files in Git. Data, reports,
checkpoints, weights, credentials and caches belong in ignored local paths.
The full course checks remain `uv run ruff check .`, `uv run ruff format --check .`
and `uv run pytest` with the unchanged 80% coverage threshold.
