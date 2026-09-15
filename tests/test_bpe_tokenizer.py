from src.BPEtokenizer import BPE_Algorithm, BPE_Encoder, BPE_Vocabulary


def test_pretokenize():
    tokenizer = BPE_Algorithm(
        training_text="This is a test to see if they'll be able to pass. I'm not sure.",
        merge_size=0,
    )

    result = tokenizer.pretokenize()

    assert result == [
        ("this", 1),
        (" is", 1),
        (" a", 1),
        (" test", 1),
        (" to", 2),
        (" see", 1),
        (" if", 1),
        (" they", 1),
        ("'ll", 1),
        (" be", 1),
        (" able", 1),
        (" pass", 1),
        (".", 2),
        (" i", 1),
        ("'m", 1),
        (" not", 1),
        (" sure", 1),
    ]


def test_base_vocab():
    tokenizer = BPE_Algorithm(training_text=" banana band banana", merge_size=0)

    result = tokenizer.base_vocab(tokenizer.pretokenize())

    assert tokenizer.bpe_merges == [" ", "b", "a", "n", "d"]
    assert result == [
        ([" ", "b", "a", "n", "a", "n", "a"], 2),
        ([" ", "b", "a", "n", "d"], 1),
    ]


def test_merge_pairs():
    tokenizer = BPE_Algorithm(training_text="", merge_size=0)

    result = tokenizer.merge_pairs(
        [
            (["h", "u", "g"], 10),
            (["p", "u", "g"], 5),
            (["b", "u", "g"], 3),
            (["u", "g"], 2),
        ]
    )

    assert result == [
        (["h", "ug"], 10),
        (["p", "ug"], 5),
        (["b", "ug"], 3),
        (["ug"], 2),
    ]
    assert tokenizer.bpe_merges == ["ug"]

    second_tokenizer = BPE_Algorithm(training_text="", merge_size=0)
    second_result = second_tokenizer.merge_pairs(
        [
            (["a", "b", "a"], 4),
            (["a", "b", "c"], 2),
            (["b", "a"], 1),
            (["a", "b"], 3),
        ]
    )

    assert second_result == [
        (["ab", "a"], 4),
        (["ab", "c"], 2),
        (["b", "a"], 1),
        (["ab"], 3),
    ]
    assert second_tokenizer.bpe_merges == ["ab"]


def test_mergify_no_pairs():
    tokenizer = BPE_Algorithm(training_text="a", merge_size=2)

    tokenizer.mergify()

    assert tokenizer.bpe_merges == ["a"]


def test_mergify_one_merge():
    tokenizer = BPE_Algorithm(training_text="hug pug", merge_size=6)

    tokenizer.mergify()

    assert tokenizer.bpe_merges == ["h", "u", "g", " ", "p", "ug"]


def test_mergify_multiple_merges():
    tokenizer = BPE_Algorithm(
        training_text="I'm a programmer, and I'll program.", merge_size=20
    )

    tokenizer.mergify()

    assert tokenizer.bpe_merges == [
        "i",
        "'",
        "m",
        " ",
        "a",
        "p",
        "r",
        "o",
        "g",
        "e",
        ",",
        "n",
        "d",
        "l",
        ".",
        " a",
        " p",
        " pr",
        " pro",
        " prog",
    ]


def test_encode_decode():
    vocabulary = BPE_Vocabulary(
        [
            "i",
            "'",
            "m",
            " ",
            "a",
            "p",
            "r",
            "o",
            "g",
            "e",
            ".",
            "'m",
            "program",
            "programmer",
        ]
    )
    encoder = BPE_Encoder(vocabulary.str2id)

    token_ids = encoder.encode("I'm a programmer.x")

    assert vocabulary.str2id["i"] == 0
    assert vocabulary.str2id["programmer"] == 13
    assert vocabulary.str2id["<|unk|>"] == 14
    assert token_ids == [0, 11, 3, 4, 3, 13, 10, 14]
    assert encoder.decode(token_ids) == "i'm a programmer.<|unk|>"
    assert encoder.get_pairs("HUG") == [("h", "u"), ("u", "g")]


def test_pipeline():
    text = "I'm a programmer, and I'll program."
    algorithm = BPE_Algorithm(training_text=text, merge_size=20)

    algorithm.mergify()
    vocabulary = BPE_Vocabulary(algorithm.bpe_merges)
    encoder = BPE_Encoder(vocabulary.str2id)
    token_ids = encoder.encode(text)

    assert len(algorithm.bpe_merges) == 20
    assert all(token_id in vocabulary.id2str for token_id in token_ids)
    assert encoder.decode(token_ids) == text.lower()
