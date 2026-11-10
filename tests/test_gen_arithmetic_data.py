import re

import pytest

from src.gen_arithmetic_data import generate_arithmetic_dataset


def test_dataset_is_reproducible_for_same_seed():
    first_dataset = generate_arithmetic_dataset(count=10, seed=123)
    second_dataset = generate_arithmetic_dataset(count=10, seed=123)

    assert first_dataset == second_dataset


def test_generated_answers_are_correct_with_different_seed():
    dataset = generate_arithmetic_dataset(count=10, seed=456)

    assert len(dataset) == 10
    for example in dataset:
        match = re.fullmatch(
            r"What is (\d{5}) \+ (\d{5}) - (\d{4})\?", example["question"]
        )
        assert match is not None
        number1, number2, number3 = map(int, match.groups())
        assert example["answer"] == str(number1 + number2 - number3)


@pytest.mark.parametrize("count", [0, -1, -10])
def test_non_positive_count_raises_value_error(count):
    with pytest.raises(ValueError, match="count must be positive"):
        generate_arithmetic_dataset(count=count)
