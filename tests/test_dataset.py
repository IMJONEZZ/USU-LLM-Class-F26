"""Tests for the training data builder.

The one I care about most is `test_every_counting_answer_is_actually_right`.
The whole argument for generating the counting slice in Python instead of
asking a model for it is that the answers can't be wrong, so that claim should
have a test behind it rather than just being asserted in my plan.

The rest are mostly about leakage. I'm not allowed to train on the ten graded
questions and the check that enforces that is worth more than the data.
"""

import re

import pytest

from src.dataset import (
    BANNED_COUNTING_WORDS,
    FAMILIES,
    HOMEWORK_QUESTIONS,
    balance_by_source,
    build_dataset,
    build_letter_counting,
    build_trick_questions,
    check_leakage,
    clean_paraphrase,
    load_chat_rows,
    paraphrase_is_safe,
    paraphrase_rows,
    similarity,
    to_messages,
)

# --- the leakage guard --------------------------------------------------------


@pytest.mark.parametrize("graded", HOMEWORK_QUESTIONS)
def test_guard_catches_every_graded_question_verbatim(graded):
    with pytest.raises(ValueError, match="too close to a graded question"):
        check_leakage([{"question": graded}])


def test_guard_catches_a_graded_question_with_the_numbers_changed():
    # This is the shape I was actually worried about. Swapping 600 for 900
    # doesn't make it a different question.
    sneaky = "If it takes 1 hour for 60 people to play an Opera, how many hours will it take 900 people to play the same opera?"
    with pytest.raises(ValueError):
        check_leakage([{"question": sneaky}])


def test_guard_lets_an_unrelated_question_through():
    check_leakage([{"question": "How do I boil an egg?"}])


def test_similarity_is_one_for_identical_text():
    assert similarity("how many Ls in parallel", "how many Ls in parallel") == 1.0


def test_similarity_ignores_stopwords():
    # Padding one side with "what is the" and "of the" shouldn't move the score.
    assert similarity("riddle sphinx", "riddle sphinx") == 1.0
    assert similarity("what is the riddle of the sphinx", "riddle sphinx") == 1.0


def test_two_short_questions_sharing_their_only_word_look_identical():
    """A known false positive, and the right way round.

    Both of these come down to the single word "egg" once the stopwords are
    gone, so the guard calls them the same question. I'd rather it refuse a
    harmless row than let a graded one through, so I'm leaving it.
    """
    assert similarity("what is an egg", "how do I do the egg") == 1.0


def test_similarity_of_nothing_is_zero():
    assert similarity("", "anything at all") == 0.0


# --- the trick questions ------------------------------------------------------


def test_every_family_contributes_rows():
    rows = build_trick_questions()
    produced = {row["source"] for row in rows}
    assert produced == {family["name"] for family in FAMILIES}


def test_trick_rows_are_every_template_crossed_with_every_slot():
    expected = sum(len(f["templates"]) * len(f["slots"]) for f in FAMILIES)
    assert len(build_trick_questions()) == expected


def test_no_template_slot_was_left_unfilled():
    # A typo in a slot name shows up as a leftover {brace} in the output.
    for row in build_trick_questions():
        assert "{" not in row["question"]
        assert "{" not in row["answer"]


def test_trick_questions_do_not_leak():
    check_leakage(build_trick_questions())


def test_shuffling_is_seeded():
    assert build_trick_questions(seed=1) == build_trick_questions(seed=1)
    assert build_trick_questions(seed=1) != build_trick_questions(seed=2)


# --- letter counting ----------------------------------------------------------


def test_every_counting_answer_is_actually_right():
    """Re-derive the count from the question and check the answer agrees."""
    for row in build_letter_counting(limit=10_000):
        letter, word = re.match(
            r"How many (\w)s are in the word (\w+)\?", row["question"]
        ).groups()
        truth = word.count(letter.lower())
        claimed = int(re.search(r"so there (?:is|are) (\d+) ", row["answer"]).group(1))
        assert claimed == truth, row


def test_counting_spells_the_word_before_it_commits_to_a_number():
    """Order matters here and it cost me a run to find out.

    With the count first the model had to pick a number before it had looked
    at the letters, and it answered 2 for "parallel". Spelling first and
    counting afterwards got it right.
    """
    for row in build_letter_counting(limit=60):
        answer = row["answer"]
        assert answer.startswith("Spelling it out:")
        assert answer.index("Spelling") < answer.index("so there")


def test_counting_answers_spell_the_word_out_first():
    for row in build_letter_counting(limit=50):
        word = re.search(r"word (\w+)\?", row["question"]).group(1)
        assert "-".join(word) in row["answer"]


def test_the_graded_word_never_appears():
    rows = build_letter_counting(limit=10_000)
    assert not any("parallel" in row["question"].lower() for row in rows)


def test_banned_words_are_matched_as_substrings():
    # "parallelogram" contains the word I'm graded on, so it has to go too.
    rows = build_letter_counting(words=["parallelogram", "butter"], limit=100)
    assert {re.search(r"word (\w+)\?", r["question"]).group(1) for r in rows} == {
        "butter"
    }


def test_banned_list_still_holds_the_graded_word():
    assert "parallel" in BANNED_COUNTING_WORDS


def test_counting_is_not_mostly_the_answer_one():
    """A lopsided slice teaches it to say "one" and stop thinking."""
    rows = build_letter_counting(limit=240)
    ones = sum(1 for r in rows if "so there is 1 " in r["answer"])
    assert ones / len(rows) < 0.5


def test_counting_respects_the_limit():
    assert len(build_letter_counting(limit=12)) <= 12


# --- paraphrases --------------------------------------------------------------


def test_clean_paraphrase_strips_the_chatty_preamble():
    raw = 'Sure! Here you go:\n\n"How long would 110 players need?"\n\nHope that helps!'
    assert clean_paraphrase(raw) == "How long would 110 players need?"


def test_clean_paraphrase_drops_list_numbering():
    assert clean_paraphrase("1. How many Rs are in the word butter?") == (
        "How many Rs are in the word butter?"
    )


def test_clean_paraphrase_returns_nothing_when_there_is_no_question():
    assert clean_paraphrase("I'm not sure what you mean.") is None
    assert clean_paraphrase("") is None


def test_a_paraphrase_that_changes_a_number_is_rejected():
    # The answers were worked out from the slots, so a changed number would
    # leave the row saying something false.
    assert not paraphrase_is_safe(
        "How long do 60 people need?", "How long do 6 people need?"
    )


def test_a_paraphrase_identical_to_the_original_is_rejected():
    assert not paraphrase_is_safe("How many Rs in butter?", "How many Rs in butter?")


def test_a_real_reword_is_accepted():
    assert paraphrase_is_safe(
        "If it takes 11 players 90 minutes, how long for 110?",
        "How long do 110 players need, if 11 take 90 minutes?",
    )


def test_paraphrase_rows_keeps_the_original_answer():
    originals = [
        {"question": "How many Rs are in butter?", "answer": "Two.", "source": "x"}
    ]
    kept = paraphrase_rows(originals, ["Butter has how many Rs?"])
    assert kept[0]["answer"] == "Two."
    assert kept[0]["source"] == "x_paraphrase"


def test_paraphrase_rows_drops_the_bad_ones():
    originals = [
        {"question": "How long do 60 people need?", "answer": "1 hour.", "source": "x"},
        {"question": "How long do 60 people need?", "answer": "1 hour.", "source": "x"},
    ]
    kept = paraphrase_rows(
        originals, ["How long do 6 people need?", "no question here"]
    )
    assert kept == []


# --- assembly -----------------------------------------------------------------


def test_to_messages_puts_the_answer_in_the_assistant_turn():
    out = to_messages([{"question": "Q?", "answer": "A."}])
    assert out[0]["messages"] == [
        {"role": "user", "content": "Q?"},
        {"role": "assistant", "content": "A."},
    ]


def test_to_messages_can_prepend_a_system_prompt():
    out = to_messages([{"question": "Q?", "answer": "A."}], system_prompt="Be brief.")
    assert out[0]["messages"][0] == {"role": "system", "content": "Be brief."}


def test_build_dataset_includes_everything_it_was_given():
    chat = [{"question": "Say hi", "answer": "Hi.", "source": "chat"}]
    extra = [{"question": "A reworded one?", "answer": "Yes.", "source": "p"}]
    rows = build_dataset(chat_rows=chat, paraphrases=extra, counting_limit=10)
    sources = {row["source"] for row in rows}
    assert "chat" in sources and "p" in sources and "letter_counting" in sources


def test_build_dataset_refuses_a_leaky_chat_row():
    leaky = [
        {"question": HOMEWORK_QUESTIONS[3], "answer": "Nothing.", "source": "chat"}
    ]
    with pytest.raises(ValueError):
        build_dataset(chat_rows=leaky, counting_limit=5)


def test_build_dataset_is_reproducible():
    assert build_dataset(seed=7, counting_limit=20) == build_dataset(
        seed=7, counting_limit=20
    )


# --- balancing and the chat slice --------------------------------------------


def test_balance_caps_each_source():
    rows = [{"question": f"q{i}?", "answer": "a", "source": "wide"} for i in range(50)]
    rows += [{"question": "narrow?", "answer": "a", "source": "narrow"}]
    out = balance_by_source(rows, cap=10)
    counts = {}
    for row in out:
        counts[row["source"]] = counts.get(row["source"], 0) + 1
    assert counts == {"wide": 10, "narrow": 1}


def test_build_dataset_applies_the_cap_to_paraphrases():
    paras = [
        {
            "question": f"reworded {i}?",
            "answer": "a",
            "source": "same_duration_paraphrase",
        }
        for i in range(40)
    ]
    rows = build_dataset(paraphrases=paras, counting_limit=5, paraphrase_cap=6)
    kept = [r for r in rows if r["source"] == "same_duration_paraphrase"]
    assert len(kept) == 6


def test_chat_rows_keep_only_single_turn_exchanges():
    fake = [
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
            ]
        },
    ]
    rows = load_chat_rows(dataset=fake)
    assert len(rows) == 1
    assert rows[0] == {"question": "hi", "answer": "hello", "source": "no_robots"}


def test_chat_rows_drop_the_long_ones():
    fake = [
        {
            "messages": [
                {"role": "user", "content": "x" * 400},
                {"role": "assistant", "content": "short"},
            ]
        },
    ]
    assert load_chat_rows(dataset=fake) == []


def test_chat_rows_respect_the_limit():
    fake = [
        {
            "messages": [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": "a"},
            ]
        }
        for i in range(20)
    ]
    assert len(load_chat_rows(limit=4, dataset=fake)) == 4


def test_chat_rows_drop_a_pair_that_starts_with_the_assistant():
    # no_robots has a few rows where the roles aren't user-then-assistant, and
    # training on those would teach the model to speak first.
    fake = [
        {
            "messages": [
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "hi"},
            ]
        },
    ]
    assert load_chat_rows(dataset=fake) == []
