"""Check sequence construction separately from PyTorch batch assembly."""

# Most tests use tiny ASCII examples whose answers can be checked by hand.
# A separate integration test uses the saved BPE model and Unicode dialogue.
# These tests check training DATA, not whether a future LLM writes good answers.
from itertools import pairwise
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from src.bpe_tokenizer import BPETokenizer, load_corpus
from src.dataloader import TextDataset, create_dataloader


@pytest.fixture
def tokenizer():
    # pytest injects this fixture into tests with a parameter named tokenizer.
    # Each test gets a fresh instance, so mutations cannot leak between tests.
    # Without learned merges, ASCII characters have predictable byte IDs:
    # A -> 65, B -> 66, etc. This isolates window arithmetic from BPE training.
    return BPETokenizer()


# Each row below runs the same test with a different boundary configuration.
# Expected outputs are written out instead of recomputing the loader's formula;
# repeating that formula here could reproduce the same bug in code and test.
@pytest.mark.parametrize(
    ("text", "max_length", "stride", "expected_inputs", "expected_targets"),
    [
        # L+1 source tokens: exactly one pair, with no extra duplicate window.
        ("ABCDE", 4, None, ["ABCD"], ["BCDE"]),
        # L+2 tokens: one regular pair and a final pair shifted by just one.
        ("ABCDEF", 4, None, ["ABCD", "BCDE"], ["BCDE", "CDEF"]),
        # Starts 0 and 4 already reach I as a target: do not append a third pair.
        ("ABCDEFGHI", 4, None, ["ABCD", "EFGH"], ["BCDE", "FGHI"]),
        # Default stride gives starts 0 and 4; add start 5 to reach target J.
        (
            "ABCDEFGHIJ",
            4,
            None,
            ["ABCD", "EFGH", "FGHI"],
            ["BCDE", "FGHI", "GHIJ"],
        ),
        # Even with overlap at starts 0, 2, and 4, start 5 is still needed.
        (
            "ABCDEFGHIJ",
            4,
            2,
            ["ABCD", "CDEF", "EFGH", "FGHI"],
            ["BCDE", "DEFG", "FGHI", "GHIJ"],
        ),
        # Overlapping regular windows can also land exactly at the corpus end.
        ("ABCDEFG", 4, 2, ["ABCD", "CDEF"], ["BCDE", "DEFG"]),
        # Minimum legal length: each input and target contains one token.
        ("ABC", 1, None, ["A", "B"], ["B", "C"]),
        # Minimum stride: move one token at a time, with maximal regular overlap.
        ("ABCDEF", 4, 1, ["ABCD", "BCDE"], ["BCDE", "CDEF"]),
    ],
)
def test_exact_windows_and_tail(
    tokenizer, text, max_length, stride, expected_inputs, expected_targets
):
    dataset = TextDataset(text, tokenizer, max_length, stride)

    # The count catches missing or duplicate windows. strict=True also makes
    # zip raise if any iterable is shorter: ordinary zip would silently stop.
    assert len(dataset) == len(expected_inputs)
    for (inputs, targets), expected_input, expected_target in zip(
        dataset, expected_inputs, expected_targets, strict=True
    ):
        # Tensor.tolist() gives ordinary integers for exact comparisons.
        # str.encode() gives UTF-8 bytes; for these ASCII fixtures, those bytes
        # are the expected token IDs. We are not decoding a trained BPE here.
        assert inputs.tolist() == list(expected_input.encode())
        assert targets.tolist() == list(expected_target.encode())
        # Correct values alone would miss floating-point IDs or unintended
        # GPU placement. The loader contract is integer tensors on the CPU.
        assert inputs.dtype == targets.dtype == torch.long
        assert inputs.device.type == targets.device.type == "cpu"


@pytest.mark.parametrize("stride", [1, 2, 3, 4])
def test_every_next_token_transition_is_covered(tokenizer, stride):
    text = "ABCDEFGHIJKLMN"
    dataset = TextDataset(text, tokenizer, max_length=4, stride=stride)
    # Align each input token with its target to obtain the transitions actually
    # supplied for learning. A set ignores duplicates from deliberate overlap.
    # All source letters are distinct, so a missing position cannot be hidden
    # by the same token pair appearing elsewhere in this particular test text.
    observed = {
        pair
        for inputs, targets in dataset
        for pair in zip(inputs.tolist(), targets.tolist(), strict=True)
    }
    source = list(text.encode())
    # pairwise yields A->B, B->C, etc. from the source independently of window
    # arithmetic. Equality catches missing transitions and unexpected ones.
    assert observed == set(pairwise(source))


@pytest.mark.parametrize("text", ["", "A", "ABCD"])
def test_insufficient_data_raises_helpful_error(tokenizer, text):
    # Empty, shorter than L, and exactly L all lack a full input/target pair.
    # pytest.raises checks the exception type; match checks useful message text.
    # We expect an actionable error rather than a silently empty training run.
    with pytest.raises(ValueError, match="Need at least 5 tokens.*smaller max_length"):
        TextDataset(text, tokenizer, max_length=4)


@pytest.mark.parametrize("max_length", [0, -1, 2.5, True])
def test_invalid_max_length(tokenizer, max_length):
    # Zero, negatives, fractions, and bools are not valid token counts. Python
    # normally accepts True as an int, which is why it deserves a test case.
    with pytest.raises(ValueError, match="max_length must be a positive integer"):
        TextDataset("ABCDEFG", tokenizer, max_length=max_length)


@pytest.mark.parametrize("stride", [0, -1, 1.5, True, 5])
def test_invalid_stride(tokenizer, stride):
    # The final case, 5, is positive but exceeds L=4 and would allow gaps.
    # Match the setting name without depending on every word of the message.
    with pytest.raises(ValueError, match="stride"):
        TextDataset("ABCDEFG", tokenizer, max_length=4, stride=stride)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_invalid_batch_size(tokenizer, batch_size):
    # Validate the wrapper's own batch-size contract, including the bool case,
    # before PyTorch attempts to assemble batches.
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        create_dataloader("ABCDE", tokenizer, max_length=4, batch_size=batch_size)


def test_tokenizes_once_before_iteration(tokenizer):
    # wraps calls the real encoder while recording its calls. Fake output could
    # hide an integration issue; here we only want to observe how often it runs.
    tokenizer.encode = Mock(wraps=tokenizer.encode)
    loader = create_dataloader("ABCDEFGHIJ", tokenizer, max_length=4, batch_size=2)
    # Materializing each iteration visits every batch: two passes are two
    # epochs over the same loader. Neither should trigger another encoding.
    list(loader)
    list(loader)
    tokenizer.encode.assert_called_once_with("ABCDEFGHIJ")


def test_keeps_smaller_batch_separately_from_overlapping_tail(tokenizer):
    # Ten source tokens create THREE examples at L=4 (starts 0, 4, 5).
    # Batch size 2 should group them as 2 then 1. This distinguishes a leftover
    # batch of examples from the leftover tokens addressed by the final window.
    # Disable shuffle so expected row order is part of this test.
    loader = create_dataloader(
        "ABCDEFGHIJ", tokenizer, max_length=4, batch_size=2, shuffle=False
    )
    batches = list(loader)
    # Shape is [examples in this batch, tokens in each example]. The last batch
    # has fewer rows, not shorter sequences. Check targets separately as well.
    assert [tuple(inputs.shape) for inputs, _ in batches] == [(2, 4), (1, 4)]
    assert [tuple(targets.shape) for _, targets in batches] == [(2, 4), (1, 4)]
    # b"ABCD" is a bytes literal; list(...) exposes its ASCII byte IDs.
    # Exact values prove the final example was kept and paired correctly, not
    # merely that some tensors of the expected dimensions were returned.
    assert batches[0][0].tolist() == [list(b"ABCD"), list(b"EFGH")]
    assert batches[1][0].tolist() == [list(b"FGHI")]
    assert batches[1][1].tolist() == [list(b"GHIJ")]


def test_default_length_stride_and_batch_size(tokenizer):
    # 2305 = 9 * 256 + 1: enough for exactly nine regular examples, with the
    # extra token supplying the final target. The fixture has no learned merges,
    # so repeating A does not compress the source into fewer than 2305 tokens.
    loader = create_dataloader("A" * 2305, tokenizer, shuffle=False)
    assert loader.dataset.max_length == loader.dataset.stride == 256
    # Dataset length counts examples; loader length counts batches (8 then 1).
    assert len(loader.dataset) == 9
    assert len(loader) == 2
    assert [tuple(x.shape) for x, _ in loader] == [(8, 256), (1, 256)]


def test_shuffle_repeats_across_runs_but_advances_between_epochs(tokenizer):
    # Distinct first tokens identify examples without inspecting sampler internals.
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    # Construct independent loaders with identical data and seeds so we can
    # compare complete runs, not just repeated calls on the same generator.
    def make_loader():
        return create_dataloader(text, tokenizer, max_length=2, batch_size=4, seed=17)

    def epoch_order(loader):
        # A row's first token identifies its window in this distinct-letter
        # corpus. item() converts a scalar tensor to a Python number, and the
        # nested comprehension flattens all batches into one epoch's order.
        return [row[0].item() for inputs, _ in loader for row in inputs]

    # Snapshot the global generator as well. A private DataLoader generator
    # should leave that unrelated state untouched throughout this experiment.
    rng_before = torch.random.get_rng_state().clone()
    first = make_loader()
    run_one = [epoch_order(first), epoch_order(first)]
    second = make_loader()
    run_two = [epoch_order(second), epoch_order(second)]
    # Same setup reproduces both epochs across independent loaders.
    assert run_one == run_two
    # In this sufficiently large fixed test case, successive epochs differ.
    # This does not claim all random permutations must differ: small datasets
    # can repeat an order by chance, and one example has only one possible order.
    assert run_one[0] != run_one[1]
    # Order can change, but membership and multiplicity must stay the same.
    # Distinct first tokens also let us verify one visit per dataset example.
    assert sorted(run_one[0]) == sorted(run_one[1])
    assert len(set(run_one[0])) == len(first.dataset)
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_saved_tokenizer_and_corpus_preserve_dialogue_and_unicode(tmp_path):
    # tmp_path is a pytest-provided directory unique to this test. A tiny JSON
    # fixture avoids requiring the ignored full corpus or a network download.
    # Two speakers check dialogue boundaries; café and the emoji check UTF-8.
    corpus_path = tmp_path / "dialogue.json"
    corpus_path.write_text(
        '[{"Character": "LEIA", "Line": "Hello!"}, '
        '{"Character": "LUKE", "Line": "Hi, café🙂!"}]',
        encoding="utf-8",
    )
    # Resolve the model relative to this test file: parents[0] is tests/ and
    # parents[1] is the repository root. Use the REAL saved A1 tokenizer here.
    model_path = Path(__file__).resolve().parents[1] / "models/star_wars_4096.bpe.json"
    tokenizer = BPETokenizer.load(model_path)
    text = load_corpus(corpus_path)
    expected = tokenizer.encode(text)
    # Snapshot learned state so we can verify that preparing examples does not
    # train or mutate the tokenizer. Its keys/values here are immutable objects,
    # so shallow dictionary copies are sufficient for this comparison.
    original_merges = tokenizer.merges.copy()
    original_vocab = tokenizer.vocab.copy()

    # Choose L=N-1 so exactly one full window covers this entire fixture.
    # This makes the first input and final target boundaries explicit and lets
    # us safely reconstruct the whole Unicode text below.
    loader = create_dataloader(
        text, tokenizer, max_length=len(expected) - 1, shuffle=False
    )
    # iter creates a batch iterator; next retrieves its first batch. Index [0]
    # selects the first (and only) example from the batch dimension.
    inputs, targets = next(iter(loader))
    assert inputs[0].tolist() == expected[:-1]
    assert targets[0].tolist() == expected[1:]
    # The input omits the last source token; append the final target to recover
    # the COMPLETE encoded fixture. Individual byte-level slices can cut a
    # Unicode character in half and need not decode independently. Reconstructing
    # an arbitrary shorter window would not guarantee valid outer boundaries.
    complete_window = inputs[0].tolist() + [targets[0, -1].item()]
    assert tokenizer.decode(complete_window) == "LEIA\nHello!\nLUKE\nHi, café🙂!"
    assert tokenizer.merges == original_merges
    assert tokenizer.vocab == original_vocab


# GPU support is optional for this repository's tests. On a CPU-only machine,
# pytest reports this check as skipped rather than treating missing CUDA as a bug.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU unavailable")
def test_batch_can_be_transferred_by_training_code(tokenizer):
    # The loader must start on CPU even on a GPU-equipped machine. Simulate the
    # future caller moving a batch to CUDA, then back to CPU for exact comparison.
    # This proves transfer compatibility, not training speed or model memory fit.
    loader = create_dataloader("ABCDE", tokenizer, max_length=4, shuffle=False)
    inputs, targets = next(iter(loader))
    assert inputs.device.type == targets.device.type == "cpu"
    assert torch.equal(inputs.to("cuda").cpu(), inputs)
    assert torch.equal(targets.to("cuda").cpu(), targets)
