# USU LLM Class — Fall 2026

Homework-submission repository for USU LLM Class DSAI-5810/6810.

> Assignments and course materials will be added throughout the Fall 2026 semester.

## Contributing

### Set up your environment

Install `uv` using the instructions found here: <https://docs.astral.sh/uv/getting-started/installation/>

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

### Train and reuse the BPE tokenizer

A tokenizer trained on every `Character` and `Line` value in the full Star Wars
corpus is included at `models/star_wars_4096.bpe.json`. It contains 4,096 vocabulary
entries and 3,840 ordered merges. Use it immediately without the training dataset:

```bash
uv run python src/bpe_tokenizer.py --load models/star_wars_4096.bpe.json --text "Hello, café🙂!"
```

To reproduce the model, place `SW_EpisodeIV_VI.json` at the project root and run:

```bash
uv run python src/bpe_tokenizer.py --vocab-size 4096 --save models/star_wars_4096.bpe.json
```

The vocabulary size is configurable and defaults to 4,096, including the 256
initial byte tokens. Model files contain hexadecimal token bytes and ordered
merge rules in JSON. Other generated `models/*.bpe.json` files are ignored by Git.

In Python, use `tokenizer.save(path)` and `BPETokenizer.load(path)`.

### Prepare next-token training batches

Run `uv sync` to install the dependencies, including PyTorch. From a Python
session at the repository root, load the saved tokenizer and local corpus:

```python
from src.bpe_tokenizer import BPETokenizer, load_corpus
from src.dataloader import create_dataloader

tokenizer = BPETokenizer.load("models/star_wars_4096.bpe.json")
text = load_corpus("SW_EpisodeIV_VI.json")
loader = create_dataloader(text, tokenizer, max_length=256, batch_size=8, seed=0)
inputs, targets = next(iter(loader))
print(inputs.shape, targets.shape)  # Each full batch has shape [8, 256].
```

`TextDataset` encodes the text once, preserving the existing character names,
dialogue, and newlines. Each example uses 257 source tokens for 256 input tokens
and 256 targets shifted forward by one token. Set `max_length` to change this
length. Text with fewer than `max_length + 1` encoded tokens raises an error.

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
uv run ruff check
uv run ruff format --check .
uv run pytest
```

### AI assistance

The course supports Claude Code and Codex. Both assistants follow the same student-learning and academic-integrity expectations in [AGENTS.md](AGENTS.md).

### Progress

- A0 - Basic Tokenizer (done)
- A1 - BPE Tokenizer (done)
- A2 - Data Loader (done)
