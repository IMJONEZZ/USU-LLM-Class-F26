"""
Tests for the data loading/splitting functions and TextDataset/DataLoader
helpers in src/dataloader.py, updated for the new scene-based Star Wars
dataset with movie-level train/validation/test splitting.
"""

import json

import pytest

from src.bpe_tokenizer import BPETokenizer, train_bpe
from src.dataloader import (
    SPECIAL_TOKEN,
    TEST_MOVIE,
    VAL_MOVIE,
    TextDataset,
    create_dataloader,
    encode_with_special_token,
    load_scenes,
    scenes_to_text,
    split_scenes_by_movie,
)

# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_scenes_file(tmp_path):
    """A small fake scenes.jsonl file covering train/val/test movies."""
    scenes = [
        {
            "movie": "A New Hope 4K77",
            "turns": [
                {"speaker": "Luke", "text": "I want to learn the ways of the Force."},
                {"speaker": "Obi-Wan", "text": "You will, young Skywalker."},
            ],
        },
        {
            "movie": VAL_MOVIE,
            "turns": [{"speaker": "Anakin", "text": "I will do what I must."}],
        },
        {
            "movie": TEST_MOVIE,
            "turns": [
                {"speaker": "Luke", "text": "I am a Jedi, like my father before me."}
            ],
        },
        {
            "movie": "A New Hope 4K77",
            "turns": [{"speaker": "Leia", "text": "Help me, Obi-Wan Kenobi."}],
        },
    ]
    fake_file = tmp_path / "fake_scenes.jsonl"
    fake_file.write_text("\n".join(json.dumps(s) for s in scenes), encoding="utf-8")
    return str(fake_file)


@pytest.fixture
def training_text():
    """Small repetitive text so BPE has clear, predictable merges to learn."""
    return "the cat sat on the mat the cat ran the dog sat on the mat"


@pytest.fixture
def tokenizer(training_text):
    """A BPETokenizer trained on the small sample text above."""
    merges, vocab = train_bpe(training_text, num_merges=20)
    return BPETokenizer(merges, vocab)


# ---------------------------------------------------------------------------
# load_scenes() TESTS
# ---------------------------------------------------------------------------


def test_load_scenes_reads_all_scenes(fake_scenes_file):
    """load_scenes should read every scene record from the jsonl file."""
    scenes = load_scenes(fake_scenes_file)
    assert len(scenes) == 4


def test_load_scenes_preserves_movie_field(fake_scenes_file):
    """Each loaded scene should retain its movie field."""
    scenes = load_scenes(fake_scenes_file)
    movies = {scene["movie"] for scene in scenes}
    assert "A New Hope 4K77" in movies
    assert VAL_MOVIE in movies
    assert TEST_MOVIE in movies


# ---------------------------------------------------------------------------
# split_scenes_by_movie() TESTS
# ---------------------------------------------------------------------------


def test_split_scenes_by_movie_separates_correctly(fake_scenes_file):
    """Scenes should be correctly bucketed into train/val/test based on movie."""
    scenes = load_scenes(fake_scenes_file)
    train, val, test = split_scenes_by_movie(scenes)

    assert len(train) == 2
    assert len(val) == 1
    assert len(test) == 1
    assert val[0]["movie"] == VAL_MOVIE
    assert test[0]["movie"] == TEST_MOVIE


def test_split_scenes_by_movie_no_overlap(fake_scenes_file):
    """No scene should appear in more than one split -- the core anti-leakage guarantee."""
    scenes = load_scenes(fake_scenes_file)
    train, val, test = split_scenes_by_movie(scenes)

    train_ids = {id(s) for s in train}
    val_ids = {id(s) for s in val}
    test_ids = {id(s) for s in test}

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


# ---------------------------------------------------------------------------
# scenes_to_text() TESTS
# ---------------------------------------------------------------------------


def test_scenes_to_text_joins_turns_within_scene(fake_scenes_file):
    """Multiple turns within the same scene should be joined with spaces."""
    scenes = load_scenes(fake_scenes_file)
    train, _, _ = split_scenes_by_movie(scenes)
    text = scenes_to_text([train[0]])  # the scene with two turns

    assert "I want to learn the ways of the Force." in text
    assert "You will, young Skywalker." in text


def test_scenes_to_text_separates_scenes_with_special_token(fake_scenes_file):
    """Different scenes should be joined with the special end-of-text token."""
    scenes = load_scenes(fake_scenes_file)
    train, _, _ = split_scenes_by_movie(scenes)
    text = scenes_to_text(train)

    assert SPECIAL_TOKEN in text


def test_scenes_to_text_empty_list_returns_empty_string():
    """An empty list of scenes should produce an empty string."""
    assert scenes_to_text([]) == ""


# ---------------------------------------------------------------------------
# encode_with_special_token() TESTS
# ---------------------------------------------------------------------------


def test_encode_with_special_token_inserts_special_id(tokenizer):
    """The special token's ID should appear in the encoded output."""
    text = f"the cat sat {SPECIAL_TOKEN} the dog ran"
    ids = encode_with_special_token(tokenizer, text)
    special_id = tokenizer.str_to_int[SPECIAL_TOKEN]
    assert special_id in ids


def test_encode_with_special_token_does_not_split_special_token(tokenizer):
    """The special token should map to exactly one ID, never split into subwords."""
    ids = encode_with_special_token(tokenizer, SPECIAL_TOKEN)
    assert len(ids) == 1
    assert ids[0] == tokenizer.str_to_int[SPECIAL_TOKEN]


# ---------------------------------------------------------------------------
# TextDataset / create_dataloader TESTS
# ---------------------------------------------------------------------------


def test_text_dataset_target_is_input_shifted_by_one(tokenizer, training_text):
    """The target sequence should be the input sequence shifted forward by one token."""
    dataset = TextDataset(training_text, tokenizer, max_length=5, stride=2)
    input_ids, target_ids = dataset[0]

    for i in range(len(input_ids) - 1):
        assert target_ids[i].item() == input_ids[i + 1].item()


def test_create_dataloader_returns_correct_batch_shape(tokenizer, training_text):
    """The DataLoader should yield batches with the requested batch_size and max_length."""
    dataloader = create_dataloader(
        training_text,
        tokenizer,
        batch_size=2,
        max_length=5,
        stride=2,
        shuffle=False,
        drop_last=False,
    )
    inputs, targets = next(iter(dataloader))

    assert inputs.shape[1] == 5
    assert targets.shape[1] == 5
