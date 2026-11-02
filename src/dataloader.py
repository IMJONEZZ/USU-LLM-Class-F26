"""A PyTorch data loader for next-token prediction on the Star Wars scripts.

The job here is to turn 2,523 lines of dialogue into batches of
``(input, target)`` pairs, where ``target`` is ``input`` shifted left by one
token. That shift is the whole point: at every position the model sees a token
and is asked to predict the one that follows it.

Three things in here are deliberate rather than accidental, and each is
explained where it happens:

  1. The dataset windows over a stream of token IDs, not over text. Tokenizing
     is a constructor helper, not something ``__getitem__`` does. See
     :meth:`SlidingWindowDataset.from_text`.
  2. ``stride`` defaults to ``max_length`` so windows are disjoint. Overlapping
     windows are legal but have to be asked for, because they silently show the
     same token to the model more than once.
  3. Train and validation are split on the *token stream*, before any windowing
     happens. See :func:`train_val_datasets` for why the other order leaks.

Run the demo from the repository root with::

    uv run python -m src.dataloader
"""

import json

import torch
from torch.utils.data import DataLoader, Dataset

from src.bpe_tokenizer import END_OF_TEXT, BPETokenizer

DEFAULT_DATASET = "SW_EpisodeIV_VI.json"

# 256 tokens is a short context by modern standards but it keeps the number of
# training chunks reasonable for a corpus this small. batch_size 8 is just a
# starting point; both are constructor arguments.
DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 8

# How much of the token stream is held back for validation.
DEFAULT_VAL_FRACTION = 0.1


def load_lines(path=DEFAULT_DATASET):
    """Read the dataset and return just the spoken lines, in script order.

    The file is a JSON list of records that each look like
    ``{"Character": "THREEPIO", "Line": "Did you hear that?"}``. Only the
    ``Line`` field is dialogue; ``Character`` is metadata, and folding speaker
    names into the training text would teach a model to emit them as if they
    were speech.
    """
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return [
        record["Line"]
        for record in records
        if isinstance(record, dict) and record.get("Line")
    ]


def join_lines(lines, separator=END_OF_TEXT):
    """Join dialogue lines into one stream with a separator between them.

    ``<|endoftext|>`` is a real single token in ``BPETokenizer`` and its
    ``encode`` keeps it whole rather than splitting it into punctuation, so it
    can be dropped straight into the text. That gives the model a consistent
    signal that one speaker has finished and another is starting.
    """
    return separator.join(lines)


class SlidingWindowDataset(Dataset):
    """Fixed-length windows over a stream of token IDs.

    This is a map-style dataset in PyTorch's terms: it implements ``__len__``
    and ``__getitem__``, so ``DataLoader`` can index it in any order and do its
    own shuffling and batching on top.

    It takes token IDs rather than text on purpose. Windowing and tokenizing
    are separate concerns, and keeping them apart is what makes it possible to
    split the token stream before windowing it (see :func:`train_val_datasets`).
    Use :meth:`from_text` when starting from a string.
    """

    def __init__(self, token_ids, max_length=DEFAULT_MAX_LENGTH, stride=None):
        if max_length < 1:
            raise ValueError(f"max_length must be at least 1, got {max_length}")

        # Disjoint windows by default. With stride < max_length the windows
        # overlap, which means a given token appears in more than one training
        # example. That is sometimes what you want, but it is data duplication
        # rather than extra data, so it should be an explicit choice.
        stride = max_length if stride is None else stride
        if stride < 1:
            raise ValueError(f"stride must be at least 1, got {stride}")

        # Tokenize-once lives at the call site; by here the IDs are already
        # computed. Holding them as a single tensor means every __getitem__ is
        # a view into this one allocation instead of building a new list.
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.max_length = max_length
        self.stride = stride

        # A window starting at i needs tokens i .. i+max_length inclusive:
        # max_length of them for the input, plus one more so the target can be
        # shifted by one. So the last legal start is len - max_length - 1, and
        # range() stopping at len - max_length gives exactly that.
        #
        # A corpus shorter than max_length + 1 therefore yields no windows at
        # all, rather than a short, silently-padded one.
        self.starts = range(0, max(0, len(self.token_ids) - max_length), stride)

    @classmethod
    def from_text(cls, text, tokenizer, max_length=DEFAULT_MAX_LENGTH, stride=None):
        """Build a dataset from raw text using any tokenizer with ``encode``.

        The tokenizer is passed in rather than constructed here. Anything with
        an ``encode(str) -> list[int]`` method works, which keeps the loader
        independent of assignment 1's BPE implementation and lets tests use a
        trivial stand-in instead of training a real vocabulary.

        Tokenizing happens once, here, rather than inside ``__getitem__``.
        ``__getitem__`` runs once per example per epoch, so tokenizing there
        would redo identical work every epoch for no benefit.
        """
        return cls(tokenizer.encode(text), max_length=max_length, stride=stride)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        """Return one ``(input, target)`` pair, target shifted left by one."""
        start = self.starts[index]
        end = start + self.max_length
        # Both are views into self.token_ids, so this copies nothing. The
        # target is the same span moved one token to the right.
        return self.token_ids[start:end], self.token_ids[start + 1 : end + 1]


def train_val_datasets(
    text,
    tokenizer,
    max_length=DEFAULT_MAX_LENGTH,
    stride=None,
    val_fraction=DEFAULT_VAL_FRACTION,
):
    """Split the token stream first, then window each side independently.

    The order matters. Windowing the whole corpus and then splitting the
    resulting chunks looks equivalent and is not: with overlapping windows, a
    chunk that lands in validation shares tokens with the chunks on either side
    of it, and those neighbours are probably in training. Validation loss then
    measures partly-memorised text.

    Splitting the token IDs at a single index and windowing the two halves
    separately makes that impossible. Training windows only ever index below
    the cut and validation windows only above it, so the window that would have
    straddled the boundary simply never gets built. The cost is one window.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")

    token_ids = tokenizer.encode(text)
    cut = int(len(token_ids) * (1.0 - val_fraction))

    train = SlidingWindowDataset(token_ids[:cut], max_length, stride)
    validation = SlidingWindowDataset(token_ids[cut:], max_length, stride)
    return train, validation


def create_dataloader(
    dataset,
    batch_size=DEFAULT_BATCH_SIZE,
    shuffle=True,
    drop_last=False,
    num_workers=0,
    seed=None,
):
    """Wrap a dataset in a ``DataLoader``.

    ``num_workers`` defaults to 0 deliberately, not by omission. The token
    stream is already in memory and ``__getitem__`` is a tensor slice, so
    worker processes would add fork and pickle overhead to something that is
    already nearly free.

    ``seed`` is the reproducibility handle: PyTorch shuffles using a
    ``Generator``, so seeding one makes the batch order repeatable, which is
    what lets the tests assert on it.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        generator=generator,
    )


def build_dataloaders(
    path=DEFAULT_DATASET,
    tokenizer=None,
    max_length=DEFAULT_MAX_LENGTH,
    stride=None,
    batch_size=DEFAULT_BATCH_SIZE,
    val_fraction=DEFAULT_VAL_FRACTION,
    vocab_size=1000,
    seed=None,
):
    """Go from the dataset file to a pair of ready-to-iterate loaders.

    If no tokenizer is supplied, a ``BPETokenizer`` is trained on the corpus
    itself. Note that this trains on the whole stream, validation included,
    which is fine for building batches but would need revisiting before the
    validation numbers were used to make claims about a model.
    """
    text = join_lines(load_lines(path))

    if tokenizer is None:
        tokenizer = BPETokenizer().train(text, vocab_size=vocab_size)

    train_set, val_set = train_val_datasets(
        text, tokenizer, max_length, stride, val_fraction
    )
    train_loader = create_dataloader(
        train_set, batch_size=batch_size, shuffle=True, seed=seed
    )
    # Validation is not shuffled: there is no gradient step, and a stable order
    # makes runs comparable to each other.
    val_loader = create_dataloader(
        val_set, batch_size=batch_size, shuffle=False, seed=seed
    )
    return train_loader, val_loader


def main():
    """Build the loaders from the real dataset and report what came out."""
    lines = load_lines()
    text = join_lines(lines)
    print(f"dialogue lines      {len(lines)}")
    print(f"characters of text  {len(text)}")

    train_loader, val_loader = build_dataloaders(seed=0)

    train_set = train_loader.dataset
    val_set = val_loader.dataset
    print(f"tokens (train)      {len(train_set.token_ids)}")
    print(f"tokens (validation) {len(val_set.token_ids)}")
    print(f"window size         {train_set.max_length}")
    print(f"stride              {train_set.stride}")
    print(f"chunks (train)      {len(train_set)}")
    print(f"chunks (validation) {len(val_set)}")
    print(f"batches (train)     {len(train_loader)}")

    inputs, targets = next(iter(train_loader))
    print(
        f"batch shapes        inputs {tuple(inputs.shape)}, targets {tuple(targets.shape)}"
    )
    print(f"target is shifted   {torch.equal(inputs[0, 1:], targets[0, :-1])}")


if __name__ == "__main__":  # pragma: no cover
    main()
