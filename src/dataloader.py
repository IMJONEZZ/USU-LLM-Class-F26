import truststore

truststore.inject_into_ssl()

from datasets import load_dataset


def load_imdb_subset(n: int = 100, seed: int = 42):
    return (
        load_dataset("stanfordnlp/imdb", split="test")
        .shuffle(seed=seed)
        .select(range(n))
    )
