# USU LLM Class — Fall 2026

Homework-submission repository for USU LLM Class DSAI-5810/6810.

> Assignments and course materials will be added throughout the Fall 2026 semester.

## Contributing

### Set up your environment

Install `uv` using the instructions found here: <https://docs.astral.sh/uv/getting-started/installation/>

The project requires Python 3.14 or newer. Run the following commands from the
repository root. Sync the project and its development dependencies:

```bash
uv sync
```

### Start experimenting

Run the template entrypoint script (it currently prints nothing and does not
train a tokenizer or language model):

```bash
uv run src/main.py
```

If you prefer a Jupyter notebook:

```bash
uv run --with jupyter jupyter lab
```

### Train and reuse the BPE tokenizer

Datasets and trained tokenizers stay local. `SW_EpisodeIV_VI.json`, the root
`data/` directory, and generated `*.bpe.json` or `bpe.json` files are ignored by Git. No pretrained
tokenizer is included in the repository.

With `SW_EpisodeIV_VI.json` at the project root, train without typing its path:

```bash
uv run python src/bpe_tokenizer.py --vocab-size 4096
```

To use another dataset, supply its path:

```bash
uv run python src/bpe_tokenizer.py data/dialogue.json --vocab-size 4096
```

Datasets must be JSON arrays with string `Character` and `Line` fields:

```json
[{"Character": "NARRATOR", "Line": "The trees are green."}]
```

The loader joins both fields from every record with newlines. Convert other
schemas to these field names before loading. The Star Wars fallback applies
only when the CLI receives no corpus path. If the selected file is missing,
Python raises `FileNotFoundError`; an explicit path does not fall back. In Python,
`load_corpus(path)` always requires a path and has no automatic fallback.

Saving is optional. With the local Star Wars dataset present, train and save
once. Later, loading the saved tokenizer needs no training dataset:

```bash
uv run python src/bpe_tokenizer.py --vocab-size 4096 --save models/star_wars_4096.bpe.json
uv run python src/bpe_tokenizer.py --load models/star_wars_4096.bpe.json --text "Hello, café🙂!"
```

The vocabulary size defaults to 4,096, including the 256 initial byte tokens.
Small corpora may finish below the requested size. Model files contain
hexadecimal token bytes and ordered merge rules in JSON. In Python, use
`tokenizer.save(path)` and `BPETokenizer.load(path)` for optional persistence.
Use a `*.bpe.json` filename to match the ignore rule. `--load` cannot be combined
with a corpus path or `--vocab-size`.

### Prepare next-token training batches

Run `uv sync` to install the dependencies, including PyTorch. From a Python
session at the repository root, train a tokenizer and prepare batches in memory:

```python
from src.bpe_tokenizer import BPETokenizer, load_corpus
from src.dataloader import create_dataloader

text = load_corpus("SW_EpisodeIV_VI.json")  # Or another JSON dataset path.
tokenizer = BPETokenizer()
tokenizer.train(text, vocab_size=4096)
loader = create_dataloader(text, tokenizer, max_length=256, batch_size=8, seed=0)
inputs, targets = next(iter(loader))
print(inputs.shape, targets.shape)  # Each full batch has shape [8, 256].
```

`TextDataset` encodes the text once, preserving the existing character names,
dialogue, and newlines. Each example uses 257 source tokens for 256 input tokens
and 256 targets shifted forward by one token. Set `max_length` to change this
length. Text with fewer than `max_length + 1` encoded tokens raises an error.
The example needs at least 257 tokens after BPE encoding; use more data or a
smaller `max_length` for small datasets. The one-record JSON above illustrates
the schema and is too small for this batching example.

`stride` defaults to `max_length`. Smaller strides enable overlapping regular
windows; values outside `1 <= stride <= max_length` raise an error. If necessary,
one additional full window ends with the final corpus token as its final target.
This final window may overlap, but it avoids padding or losing the tail. It is
not added when a regular window already reaches the end.

The loader shuffles examples by default, keeps the final smaller batch, and uses
a private random generator (default seed `0`). Recreating the loader with the
same seed reproduces its shuffle sequence in the same environment. Reusing the
loader for another epoch advances that sequence. Use `shuffle=False` to inspect
examples in corpus order. This preserves token order within each example.

Batches contain CPU `torch.long` tensors. A future training loop can select CUDA
when available and move batches to the GPU; device selection is outside the
loader. GPU execution requires a compatible PyTorch build and driver. CPU-only
machines can run the data loader and its tests.

For inspection, a complete source window can be reconstructed from one input
sequence plus its last target token. Individual byte-level windows can cut a
Unicode character in half, so not every window can be decoded independently.

### Set up pre-commit (optional)

CI runs the project checks for every push and pull request. To run those checks before committing, install the hooks:

```bash
uvx prek install
```

### Run checks locally

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Tests use small fixed corpora and temporary saved tokenizers. No local dataset
or committed BPE file is needed to run them. The CUDA test skips when unavailable.
The legacy `src/tokenizer.py` remains unchanged as the earlier assignment example.

### AI assistance

The course supports Claude Code and Codex. Both assistants follow the same student-learning and academic-integrity expectations in [AGENTS.md](AGENTS.md).

### Progress

- A0 - Basic Tokenizer (done)
- A1 - BPE Tokenizer (done)
- A2 - Data Loader (done)
