from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader

from src.dataloader import StarWarsDataset, create_dataloader
from src.tokenizer import SentencePieceTokenizer

# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def mock_tokenizer():
    """Mock SentencePieceTokenizer instance to isolate dataset logic."""
    return MagicMock(spec=SentencePieceTokenizer)


@pytest.fixture
def sample_raw_text():
    return "A long time ago in a galaxy far, far away... It is a period of civil war."


@pytest.fixture
def sample_text_list():
    return [
        "A long time ago in a galaxy far, far away...",
        "It is a period of civil war. Rebel spaceships, striking from a hidden base...",
    ]


# =====================================================================
# StarWarsDataset Tests
# =====================================================================


class TestStarWarsDataset:
    def test_sliding_window_shapes_and_alignment(self, mock_tokenizer):
        """Verify inputs and targets are offset by 1 token and match expected tensor shapes."""
        mock_tokenizer.encode.return_value = list(range(20))
        max_length = 5
        stride = 2

        dataset = StarWarsDataset(
            "dummy text", mock_tokenizer, max_length=max_length, stride=stride
        )

        # Expected indices: 0, 2, 4, 6, 8, 10, 12, 14 -> 8 windows
        assert len(dataset) == 8

        # Validate first window
        input_0, target_0 = dataset[0]
        assert torch.equal(input_0, torch.tensor([0, 1, 2, 3, 4]))
        assert torch.equal(target_0, torch.tensor([1, 2, 3, 4, 5]))
        assert input_0.dtype == torch.int64
        assert target_0.dtype == torch.int64

        # Validate second window shift by stride
        input_1, target_1 = dataset[1]
        assert torch.equal(input_1, torch.tensor([2, 3, 4, 5, 6]))
        assert torch.equal(target_1, torch.tensor([3, 4, 5, 6, 7]))

    def test_text_shorter_than_or_equal_to_max_length(self, mock_tokenizer):
        """Verify that token length <= max_length yields an empty dataset."""
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        dataset = StarWarsDataset("short text", mock_tokenizer, max_length=5, stride=1)

        assert len(dataset) == 0

    def test_non_overlapping_stride(self, mock_tokenizer):
        """Verify windowing when stride is equal to or greater than max_length."""
        mock_tokenizer.encode.return_value = list(range(30))
        max_length = 8
        stride = 8

        dataset = StarWarsDataset(
            "text", mock_tokenizer, max_length=max_length, stride=stride
        )

        # range(0, 22, 8) -> steps: 0, 8, 16 -> 3 chunks
        assert len(dataset) == 3
        assert torch.equal(dataset[0][0], torch.tensor(list(range(8))))
        assert torch.equal(dataset[1][0], torch.tensor(list(range(8, 16))))
        assert torch.equal(dataset[2][0], torch.tensor(list(range(16, 24))))

    def test_getitem_out_of_bounds(self, mock_tokenizer):
        """Verify IndexError is raised when indexing past the dataset length."""
        mock_tokenizer.encode.return_value = list(range(10))
        dataset = StarWarsDataset("text", mock_tokenizer, max_length=4, stride=2)

        with pytest.raises(IndexError):
            _ = dataset[len(dataset) + 1]


# =====================================================================
# create_dataloader Tests
# =====================================================================


class TestCreateDataLoader:
    @patch("src.dataloader.SentencePieceTokenizer")
    def test_create_dataloader_uses_defaults(self, mock_sp_cls):
        """Verify the public defaults are passed through to PyTorch's loader."""
        mock_instance = MagicMock(spec=SentencePieceTokenizer)
        # These tokens make three windows under the default 128/64 settings.
        mock_instance.encode.return_value = list(range(257))
        mock_sp_cls.return_value = mock_instance

        loader = create_dataloader("default configuration")

        assert loader.batch_size == 16
        assert loader.drop_last is True
        assert loader.num_workers == 0
        assert len(loader.dataset) == 3
        assert loader.dataset[0][0].shape == (128,)
        assert loader.dataset[1][0][0].item() == 64
        mock_instance.train.assert_called_once_with(
            "default configuration", num_merges=300
        )

    @patch("src.dataloader.SentencePieceTokenizer")
    def test_create_dataloader_with_string_input(self, mock_sp_cls, sample_raw_text):
        """Verify dataloader configuration and tokenizer method calls for string inputs."""
        mock_instance = MagicMock(spec=SentencePieceTokenizer)
        mock_instance.encode.return_value = list(range(50))
        mock_sp_cls.return_value = mock_instance

        loader = create_dataloader(
            text=sample_raw_text,
            batch_size=4,
            max_length=10,
            stride=5,
            shuffle=False,
            drop_last=False,
        )

        assert isinstance(loader, DataLoader)
        assert loader.batch_size == 4
        assert loader.drop_last is False
        mock_instance.train.assert_called_once_with(sample_raw_text, num_merges=300)
        mock_instance.encode.assert_called_once_with(sample_raw_text)

    @patch("src.dataloader.SentencePieceTokenizer")
    def test_create_dataloader_with_list_input(self, mock_sp_cls, sample_text_list):
        """Verify list inputs are concatenated with spaces before tokenizer calls."""
        mock_instance = MagicMock(spec=SentencePieceTokenizer)
        mock_instance.encode.return_value = list(range(100))
        mock_sp_cls.return_value = mock_instance

        expected_joined_text = " ".join(sample_text_list)

        loader = create_dataloader(
            text=sample_text_list, batch_size=2, max_length=8, stride=4
        )

        assert isinstance(loader, DataLoader)
        mock_instance.train.assert_called_once_with(
            expected_joined_text, num_merges=300
        )
        mock_instance.encode.assert_called_once_with(expected_joined_text)

    @patch("src.dataloader.SentencePieceTokenizer")
    def test_dataloader_batch_iteration(self, mock_sp_cls):
        """Verify DataLoader yields batched PyTorch tensors with matching dimensions."""
        mock_instance = MagicMock(spec=SentencePieceTokenizer)
        mock_instance.encode.return_value = list(range(100))
        mock_sp_cls.return_value = mock_instance

        batch_size = 4
        max_len = 16
        loader = create_dataloader(
            text="Any text",
            batch_size=batch_size,
            max_length=max_len,
            stride=8,
            shuffle=False,
            drop_last=True,
        )

        batch_inputs, batch_targets = next(iter(loader))

        assert batch_inputs.shape == (batch_size, max_len)
        assert batch_targets.shape == (batch_size, max_len)
        assert torch.equal(batch_targets[:, :-1], batch_inputs[:, 1:])

    @pytest.mark.parametrize(
        ("drop_last", "expected_batches", "expected_final_batch_size"),
        [(True, 1, 3), (False, 2, 1)],
    )
    @patch("src.dataloader.SentencePieceTokenizer")
    def test_incomplete_final_batch_respects_drop_last(
        self,
        mock_sp_cls,
        drop_last,
        expected_batches,
        expected_final_batch_size,
    ):
        """An incomplete final batch is dropped or retained without bad indexing."""
        mock_instance = MagicMock(spec=SentencePieceTokenizer)
        # With max_length=4 and stride=4, 17 tokens make four windows.
        mock_instance.encode.return_value = list(range(17))
        mock_sp_cls.return_value = mock_instance

        loader = create_dataloader(
            "four windows",
            batch_size=3,
            max_length=4,
            stride=4,
            shuffle=False,
            drop_last=drop_last,
        )
        batches = list(loader)

        assert len(batches) == expected_batches
        assert batches[-1][0].shape == (expected_final_batch_size, 4)

    @pytest.mark.parametrize("drop_last", [True, False])
    @patch("src.dataloader.SentencePieceTokenizer")
    def test_text_shorter_than_max_length_produces_no_batches(
        self, mock_sp_cls, drop_last
    ):
        """Neither batching mode can create a batch when no window exists."""
        mock_instance = MagicMock(spec=SentencePieceTokenizer)
        mock_instance.encode.return_value = [10, 11, 12]
        mock_sp_cls.return_value = mock_instance

        loader = create_dataloader(
            "short",
            batch_size=2,
            max_length=4,
            stride=1,
            shuffle=False,
            drop_last=drop_last,
        )

        assert len(loader.dataset) == 0
        assert list(loader) == []
