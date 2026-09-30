"""Prepare next-token examples and reproducible batches from tokenized text."""

# A1 converts text into IDs. This module arranges those IDs into examples;
# it does not train a tokenizer or a language model.
import torch

# Dataset supplies the interface for retrieving ONE example by index.
# DataLoader retrieves examples and stacks them into batches for iteration.
from torch.utils.data import DataLoader, Dataset

from src.bpe_tokenizer import BPETokenizer


def _require_positive_integer(name: str, value: int) -> None:
    # Type hints do not enforce input types at runtime, so validate explicitly.
    # bool needs its own rejection: Python considers True an int with value 1.
    # The underscore marks this as a helper internal to this module.
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


class TextDataset(Dataset):
    """Encode text once and retrieve fixed-length input/target pairs on the CPU.

    Regular input windows do not overlap when stride is omitted. A final
    window may overlap to retain the tail without padding. Returned tensors
    are views of the stored tokens and should not be modified in place.
    """

    def __init__(
        self,
        text: str,
        tokenizer: BPETokenizer,
        max_length: int = 256,
        stride: int | None = None,
    ):
        # max_length counts TOKENS per example, not characters or words.
        # Validate before encoding so invalid settings fail without extra work.
        _require_positive_integer("max_length", max_length)
        # None means "follow the chosen max_length." For example, changing
        # max_length to 128 also changes the default stride to 128, not 256.
        if stride is None:
            stride = max_length
        _require_positive_integer("stride", stride)
        # stride is the distance between successive regular starts:
        # smaller than max_length -> overlap; equal -> disjoint input windows.
        # Larger strides can skip next-token transitions in the corpus.
        if stride > max_length:
            raise ValueError("stride must not exceed max_length; this would skip data")

        # Encoding applies the supplied tokenizer's existing vocabulary and
        # merges; it does not retrain it. Preserve the text exactly as supplied,
        # including the existing corpus reader's newlines and dialogue fields.
        # This runs ONCE per dataset, not once per example or training epoch.
        token_ids = tokenizer.encode(text)
        # A length-4 example needs 5 source tokens:
        # source ABCDE -> input ABCD, target BCDE. E is D's next-token answer.
        # Count after BPE encoding because text length and token count differ.
        if len(token_ids) < max_length + 1:
            raise ValueError(
                f"Need at least {max_length + 1} tokens for max_length={max_length}; "
                f"got {len(token_ids)}. Use more text or a smaller max_length."
            )

        self.max_length = max_length
        self.stride = stride
        # Store the encoded corpus once, rather than copying every overlapping
        # window. torch.long is int64: token IDs are integer vocabulary indices.
        # A Python list can become a tensor directly; NumPy is not needed here.
        # The future training loop decides whether to move batches to a GPU.
        self.token_ids = torch.tensor(token_ids, dtype=torch.long, device="cpu")

        # Let N = token count and L = max_length. A pair starting at s has
        # final target index s + L. The final corpus index is N - 1, so the
        # latest valid start is s = N - L - 1 (zero-based indexing).
        last_start = len(token_ids) - max_length - 1
        # range excludes its stop: +1 lets us include last_start when a regular
        # stride lands on it. The earlier length check ensures start 0 is valid,
        # so this list always contains at least one position.
        self.starts = list(range(0, last_start + 1, stride))
        # Example: N=10, L=4, stride=4 gives regular starts [0, 4], but the
        # latest valid start is 5. Append 5 to include the last token as a target.
        # This final window may overlap; that is our chosen cost of avoiding
        # padding or unused tail tokens. Do not append a duplicate when the
        # regular windows already reach the end, including the one-window case.
        if self.starts[-1] != last_start:
            self.starts.append(last_start)

    def __len__(self) -> int:
        # len(dataset) counts examples, not tokens or batches. Each stored start
        # identifies exactly one input/target pair that DataLoader can request.
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # dataset[index] asks for an EXAMPLE index; look up its token offset.
        # Normal list indexing also supports -1 for the final example and raises
        # IndexError for an out-of-range request.
        start = self.starts[index]
        end = start + self.max_length
        # Slices exclude the end position, so each slice contains max_length
        # tokens. Shifting both boundaries by one aligns each input position
        # with its next-token target. These are tensor views sharing storage;
        # callers should clone a sample first if they need to modify it.
        return self.token_ids[start:end], self.token_ids[start + 1 : end + 1]


def create_dataloader(
    text: str,
    tokenizer: BPETokenizer,
    max_length: int = 256,
    stride: int | None = None,
    batch_size: int = 8,
    shuffle: bool = True,
    seed: int = 0,
) -> DataLoader:
    """Build a loader that retains the final smaller batch.

    A private generator makes shuffle order reproducible across newly created
    loaders with the same seed. Its state advances between epochs, producing
    fresh orders without resetting the seed or changing the global RNG state.
    Device selection and GPU transfers belong to the future training loop.
    """
    # A factory configures PyTorch's existing DataLoader rather than writing
    # custom batching logic. Validate batch size before constructing the dataset.
    _require_positive_integer("batch_size", batch_size)
    dataset = TextDataset(text, tokenizer, max_length=max_length, stride=stride)
    # Give this loader its own random state. Unrelated use of PyTorch's global
    # generator will not consume its shuffle sequence. Seed once at creation:
    # reuse the loader for subsequent epochs to let its state advance.
    # Recreating it with the same seed reproduces the order in the same setup.
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        # PyTorch's default collation stacks corresponding sample tensors.
        # With our defaults, inputs and targets each have shape [8, 256].
        batch_size=batch_size,
        # Shuffle whole examples, never the token order inside an example.
        # False is useful when inspecting windows in chronological order.
        shuffle=shuffle,
        # Keep a final batch with fewer examples (e.g. 8, 8, 3). This is separate
        # from handling leftover source tokens with the final overlapping window.
        drop_last=False,
        # Pass the private seeded generator into PyTorch's sampling machinery.
        generator=generator,
        # Retrieve slices in the calling process; no extra loader processes.
        # This does not prevent the training loop from doing GPU computation.
        num_workers=0,
    )
