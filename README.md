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

Tests use small fixtures and mocked services
without model downloads.

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
- A4 - MLOps (In Progress)

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

#### Status
