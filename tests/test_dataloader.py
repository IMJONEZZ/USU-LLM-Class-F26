from src.dataloader import load_imdb_subset


def test_returns_expected_length():
    subset = load_imdb_subset(n=100)
    assert len(subset) == 100


def test_same_seed_is_reproducible():
    first = load_imdb_subset(n=100, seed=42)
    second = load_imdb_subset(n=100, seed=42)
    assert first[0]["text"] == second[0]["text"]
    assert first[0]["label"] == second[0]["label"]
