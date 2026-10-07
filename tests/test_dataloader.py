import torch

from src.BPEtokenizer import BPE_Encoder, BPE_Vocabulary
from src.dataloader import TextDataset, create_dataloader_v1


def test_text_dataset_creates_overlapping_input_target_pairs():
    vocabulary = BPE_Vocabulary(["a", "b"])
    tokenizer = BPE_Encoder(vocabulary.str2id)

    dataset = TextDataset("ababababab", tokenizer, max_length=4, stride=2)

    assert len(dataset) == 3

    input_ids, target_ids = dataset[0]
    assert torch.equal(input_ids, torch.tensor([0, 1, 0, 1]))
    assert torch.equal(target_ids, torch.tensor([1, 0, 1, 0]))


def test_create_dataloader_v1_returns_batched_token_pairs():
    text = " ".join(f"word{index}" for index in range(100))

    dataloader = create_dataloader_v1(
        text,
        batch_size=4,
        max_length=8,
        stride=4,
        shuffle=False,
        drop_last=True,
    )

    input_ids, target_ids = next(iter(dataloader))
    assert isinstance(dataloader.dataset, TextDataset)
    assert input_ids.shape == (4, 8)
    assert target_ids.shape == (4, 8)


def test_create_dataloader_v1_keeps_partial_batch_when_requested():
    text = " ".join(f"word{index}" for index in range(100))

    dataloader = create_dataloader_v1(
        text,
        batch_size=4,
        max_length=8,
        stride=4,
        shuffle=False,
        drop_last=False,
    )

    assert len(dataloader) == (len(dataloader.dataset) + 3) // 4
