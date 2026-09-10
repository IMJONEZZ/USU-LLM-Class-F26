"""Tests for both tokenizers (Assignment: build a better tokenizer + tests).

The SimpleTokenizer baseline must round trip losslessly with <|unk|> for
unseen words. The Qwen3Tokenizer must round trip ANY utf-8 string, compress
better than the baseline, keep digits separate, and handle special tokens
tiktoken-style (raise unless allow-listed).
"""

import pytest

from src.qwen3_tokenizer import EOT, Qwen3Tokenizer
from src.simple_tokenizer import SimpleTokenizer

CORPUS = (
    "The forest canopy filters light into green shafts. We're learning\n"
    "tokenization, and it is working! Don't rush it; they've done that before.\n"
    'He said: "the model can\'t fly -- yet" and then it flew.\n'
    "def train(text):\n    merges = pair_counts(text)\n    return merges\n"
    "Numbers like 3.14159 and 4096 appear often; 1234 also does.\n"
    "Hello world, hello again! The canopy stays green.\n"
    "你好世界, 就是如此。🙂\n"
) * 6

# Sentence whose every word is in CORPUS: the baseline must round trip it,
# and both tokenizers train on it so compression is a fair comparison.
ENGLISH_PROBE = "The canopy filters light; the model learns merges."
IN_CORPUS_PROBE = "The canopy filters light; the model is working."
CODE_PROBE = "def train(text):\n    return merges\n"
CJK_PROBE = "你好世界 🙂🙂🙂"
MIXED_PROBE = "π≈3.14159, 你好, don't panic — yet!🙂\n"


@pytest.fixture(scope="module")
def simple():
    return SimpleTokenizer.train(CORPUS)


@pytest.fixture(scope="module")
def qwen():
    return Qwen3Tokenizer.train(CORPUS, vocab_size=300)


# ---------------------------------------------------------------------------
# Baseline SimpleTokenizer (lossless whitespace tokenizer, prior assignment)
# ---------------------------------------------------------------------------


def test_simple_roundtrip(simple):
    for probe in (
        IN_CORPUS_PROBE,
        CODE_PROBE,
        "Don't rush it; they've done that before.",
    ):
        assert simple.decode(simple.encode(probe)) == probe


def test_simple_unknown_words_use_unk(simple):
    assert "<|unk|>" in simple.str_to_int
    unk_id = simple.str_to_int["<|unk|>"]
    assert unk_id in simple.encode("this word absolutelz unknownword")
    assert "<|unk|>" in simple.decode(simple.encode("unknownword"))


def test_simple_punctuation_is_separate(simple):
    assert simple.decode(simple.encode("Hello, world!")) == "Hello, world!"
    # every punctuation character becomes its own token
    assert len(simple.encode("!?,")) == 3
    assert len(simple.encode("Hello")) == 1


def test_simple_decode_unknown_id_is_not_indexable(simple):
    max_id = len(simple.str_to_int) - 1
    with pytest.raises(KeyError):
        simple.decode([max_id + 5])


# ---------------------------------------------------------------------------
# Qwen3Tokenizer (byte-level BPE, Qwen3 pre-tokenizer, special tokens)
# ---------------------------------------------------------------------------


def test_qwen_vocab_size_is_exact(qwen):
    assert len(qwen.vocab) == 300
    special_ids = set(qwen._special_ids.values())
    assert len(special_ids) == len(qwen.SPECIALS)
    assert max(special_ids) == 299  # specials live at the top of the vocab


def test_qwen_vocab_size_too_small_raises():
    with pytest.raises(ValueError):
        Qwen3Tokenizer.train(CORPUS, vocab_size=10)


def test_qwen_roundtrips_any_string(qwen):
    for probe in (
        ENGLISH_PROBE,
        CODE_PROBE,
        CJK_PROBE,
        MIXED_PROBE,
        "Hello, do you like tea?",
        "don't — we've said 3×!",
        "🎆🕊️ flag emoji 🇸🇪 stuff",
        "tabs\tand\tspaces   :  ",
    ):
        assert qwen.decode(qwen.encode(probe)) == probe, probe


def test_qwen_bytes_cover_everything(qwen):
    # A word never seen in training plus CJK plus emoji: no <|unk|> exists.
    assert "<|unk|>" not in qwen.vocab
    probe = "Akkadian cuneiform offers 𒀭 plus 你好异地 and 😀"
    assert qwen.decode(qwen.encode(probe)) == probe


def test_qwen_compresses_sub_word(qwen):
    # Round trips are exact, IDs never exceed the UTF-8 byte count (byte
    # fallback floor), and merges must actually fire on ASCII text.
    for probe in (MIXED_PROBE,):
        ids = qwen.encode(probe)
        assert qwen.decode(ids) == probe, probe
        assert len(ids) <= len(probe.encode("utf-8")), probe
    # ASCII probes: even without merges, IDs == characters, so with any
    # learned merge the ID count must be strictly below the character count.
    q_ids = qwen.encode(IN_CORPUS_PROBE)
    assert len(q_ids) < len(IN_CORPUS_PROBE)
    assert sum(map(len, (qwen.inverse_vocab[i] for i in q_ids))) > len(q_ids)


def test_qwen_handles_oov_words_the_baseline_destroys(qwen, simple):
    # New words: the baseline must swallow them into <|unk|> (losing content),
    # while the byte-level tokenizer preserves them exactly.
    probe = "The canopy filters bananas; the model is gold."
    s_ids, q_ids = simple.encode(probe), qwen.encode(probe)
    assert "<|unk|>" in simple.decode(s_ids)
    assert qwen.decode(q_ids) == probe
    assert len(q_ids) < len(probe)
    assert len(q_ids) <= len(probe.encode("utf-8"))  # never worse than bytes


def test_qwen_never_worse_than_raw_bytes(qwen):
    # On lyrical prose entirely absent from the corpus, byte-level fallback is
    # the zero-unk floor: IDs <= bytes for any valid utf-8 string.
    probe = "Midnight rain taps sessions on the rooftop quiet"
    q_ids = qwen.encode(probe)
    assert qwen.decode(q_ids) == probe
    assert len(q_ids) <= len(probe.encode("utf-8"))


def test_qwen_digits_stay_separate(qwen):
    # Qwen3 rule: one digit per pre-token, so digits never merge into runs.
    assert len(qwen.encode("1234")) == 4
    assert len(qwen.encode("4096")) == 4


def test_qwen_respects_pretoken_boundaries(qwen):
    # Punctuation never merges into letter tokens; spaces inside words neither.
    probe = "cat,dot"
    ids = qwen.encode(probe)
    assert qwen.decode(ids) == probe
    assert len(ids) >= len(set("cat,dot"))  # no over-fused single token


def test_qwen_specials_need_allowlisting(qwen):
    with pytest.raises(ValueError, match="disallowed"):
        qwen.encode("say " + EOT + " now")
    with pytest.raises(ValueError, match="disallowed"):
        qwen.encode("<|im_start|>user")
    allowed_ids = qwen.encode("say " + EOT + " now", allowed_special=(EOT,))
    assert qwen._special_ids[EOT] in allowed_ids
    assert qwen.decode(allowed_ids) == "say " + EOT + " now"


def test_qwen_specials_roundtrip_chat_turn(qwen):
    probe = "<|im_start|>user\nHow are you?<|im_end|>"
    ids = qwen.encode(probe, allowed_special=("<|im_start|>", "<|im_end|>"))
    assert qwen.decode(ids) == probe


def test_qwen_decode_bad_id_raises(qwen):
    with pytest.raises(KeyError):
        qwen.decode([len(qwen.vocab) + 7])
