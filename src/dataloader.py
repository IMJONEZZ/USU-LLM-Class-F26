"""Load, group, and tokenize the Star Wars dialogue corpus."""

from collections.abc import Iterable, Mapping

DATASET_NAME = "IMJONEZZ/star-wars-dataset"
VALIDATION_FILM = "ep5_empire_strikes_back"
TEST_FILM = "ep6_return_of_the_jedi"


def combine_consecutive_lines(
    rows: Iterable[Mapping[str, object]],
) -> list[dict[str, str]]:
    """Join adjacent lines from the same speaker and film, preserving row order."""
    combined: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    for row in rows:
        film = row.get("film")
        speaker = row.get("speaker")
        text = row.get("text")
        if not isinstance(film, str) or not isinstance(text, str) or not text.strip():
            current = None
            continue

        can_join = (
            current is not None
            and isinstance(speaker, str)
            and bool(speaker.strip())
            and current["speaker"] == str(speaker)
            and current["film"] == film
        )
        if can_join:
            current["text"] = f"{current['text']} {text.strip()}"
        else:
            current = {
                "film": film,
                "speaker": "" if speaker is None else str(speaker),
                "text": text.strip(),
            }
            combined.append(current)

    return combined


def _make_lm_dataset(texts: list[str], tokenizer, block_size: int, dataset_type):
    """Tokenize dialogue and split it into fixed-size causal-LM examples."""
    examples: list[list[int]] = []
    eos_id = tokenizer.eos_token_id

    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if eos_id is not None:
            token_ids.append(eos_id)

        for start in range(0, len(token_ids), block_size):
            block = token_ids[start : start + block_size]
            if len(block) > 1:
                examples.append(block)

    return dataset_type.from_dict({"input_ids": examples})


def create_datasets(
    tokenizer,
    dataset_name: str = DATASET_NAME,
    block_size: int = 256,
) -> dict[str, object]:
    """Return tokenized train, validation, and test datasets.

    Training uses every film other than Episodes V and VI. Those two films are
    held out as validation and test data respectively.
    """
    if block_size < 2:
        raise ValueError("block_size must be at least 2 tokens")

    from datasets import Dataset, load_dataset

    source = load_dataset(dataset_name, "cues", split="train")
    grouped = combine_consecutive_lines(source)
    texts_by_split = {
        "train": [
            row["text"]
            for row in grouped
            if row["film"] not in {VALIDATION_FILM, TEST_FILM}
        ],
        "validation": [
            row["text"] for row in grouped if row["film"] == VALIDATION_FILM
        ],
        "test": [row["text"] for row in grouped if row["film"] == TEST_FILM],
    }
    return {
        split: _make_lm_dataset(texts, tokenizer, block_size, Dataset)
        for split, texts in texts_by_split.items()
    }
