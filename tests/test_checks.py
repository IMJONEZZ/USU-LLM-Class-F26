import pytest

from src import checks

# (instruction id, real-shaped kwargs, response, does Google's strict check pass it)
CASES = [
    ("punctuation:no_comma", {}, "No commas in this one.", True),
    ("punctuation:no_comma", {}, "Yes, a comma.", False),
    ("change_case:english_lowercase", {}, "all lowercase here", True),
    ("change_case:english_lowercase", {}, "Not all lowercase", False),
    ("change_case:english_capital", {}, "ALL CAPS HERE", True),
    ("change_case:english_capital", {}, "Not All Caps", False),
    (
        "length_constraints:number_sentences",
        {"num_sentences": 2, "relation": "at least"},
        "One. Two. Three.",
        True,
    ),
    (
        "length_constraints:number_sentences",
        {"num_sentences": 2, "relation": "at least"},
        "Only one sentence.",
        False,
    ),
    (
        "length_constraints:number_words",
        {"num_words": 5, "relation": "at least"},
        "one two three four five six",
        True,
    ),
    (
        "length_constraints:number_words",
        {"num_words": 5, "relation": "at least"},
        "one two",
        False,
    ),
    ("keywords:forbidden_words", {"forbidden_words": ["bad"]}, "A good day.", True),
    ("keywords:forbidden_words", {"forbidden_words": ["bad"]}, "A bad day.", False),
    (
        "startend:end_checker",
        {"end_phrase": "Is there anything else I can help with?"},
        "Done. Is there anything else I can help with?",
        True,
    ),
    (
        "startend:end_checker",
        {"end_phrase": "Is there anything else I can help with?"},
        "Done.",
        False,
    ),
    ("detectable_format:title", {}, "<<A title>>\nSome body.", True),
    ("detectable_format:title", {}, "No title here.", False),
    (
        "change_case:capital_word_frequency",
        {"capital_relation": "at least", "capital_frequency": 2},
        "THIS is ALL we need",
        True,
    ),
    (
        "change_case:capital_word_frequency",
        {"capital_relation": "at least", "capital_frequency": 2},
        "nothing in capitals here",
        False,
    ),
    ("language:response_language", {"language": "en"}, "The response is English", True),
    (
        "language:response_language",
        {"language": "en"},
        "Desayunamos en casa hoy con mucho gusto",
        False,
    ),
]


def make_example(instruction_id, kwargs):
    return {
        "key": 1,
        "prompt": "a prompt",
        "instruction_id": instruction_id,
        "kwargs": kwargs,
    }


@pytest.mark.parametrize(("instruction_id", "kwargs", "response", "expected"), CASES)
def test_passes_gives_the_official_verdict_for_the_kwargs_it_was_given(
    instruction_id, kwargs, response, expected
):
    example = make_example(instruction_id, kwargs)
    assert checks.passes(example, response) is expected


@pytest.mark.parametrize(("instruction_id", "kwargs", "response", "expected"), CASES)
def test_feedback_exists_only_when_the_check_fails_and_names_the_instruction(
    instruction_id, kwargs, response, expected
):
    feedback = checks.feedback(make_example(instruction_id, kwargs), response)
    assert (feedback is None) is expected
    if not expected:
        assert instruction_id in feedback


@pytest.mark.parametrize("response", ["", "   \n"])
def test_empty_and_whitespace_only_responses_fail_even_when_nothing_is_forbidden(
    response,
):
    # no_comma would pass an empty string on its own; the strict check's strip rule is what fails it
    example = make_example("punctuation:no_comma", {})
    assert checks.passes(example, response) is False
    assert "punctuation:no_comma" in checks.feedback(example, response)
