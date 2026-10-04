from contextlib import nullcontext

import pytest

from src.evaluator import (
    INVALID_LABEL,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    Evaluator,
    LlamaSST2Predictor,
    bootstrap_accuracy_interval,
    calculate_metrics,
    parse_sentiment_label,
    sentiment_messages,
    validate_id2label,
)


class FakeTensor:
    def __init__(self, *, width=None):
        self.shape = (2, width) if width is not None else None
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def __getitem__(self, key):
        return key


class FakeLlamaTokenizer:
    eos_token_id = 128009

    def __init__(self):
        self.padding_side = "right"
        self.pad_token_id = None
        self.conversations = None

    def apply_chat_template(self, conversations, **kwargs):
        self.conversations = conversations
        self.template_kwargs = kwargs
        return {
            "input_ids": FakeTensor(width=5),
            "attention_mask": FakeTensor(width=5),
        }

    def batch_decode(self, generated, **kwargs):
        self.decoded_slice = generated
        self.decode_kwargs = kwargs
        return ["positive", "extra explanation"]


class FakeLlamaModel:
    device = "cuda:0"

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return FakeTensor()


def test_evaluator_uses_injected_predictor():
    received_texts = []

    def fake_predictor(texts):
        received_texts.extend(texts)
        return [NEGATIVE_LABEL, POSITIVE_LABEL]

    result = Evaluator(fake_predictor, n_resamples=100).evaluate(
        ["bad", "good"],
        [NEGATIVE_LABEL, POSITIVE_LABEL],
    )

    assert received_texts == ["bad", "good"]
    assert result.accuracy == 1.0
    assert result.macro_f1 == 1.0
    assert result.quality_bar_met


def test_metrics_match_controlled_confusion_matrix():
    # Rows are true labels; columns are negative, positive, and invalid.
    result = calculate_metrics(
        labels=[0, 0, 0, 1, 1, 1],
        predictions=[0, 0, 1, 0, 1, 1],
        n_resamples=200,
    )

    assert result.example_count == 6
    assert result.confusion_matrix == ((2, 1, 0), (1, 2, 0))
    assert result.invalid_prediction_count == 0
    assert result.invalid_prediction_rate == 0.0
    assert result.accuracy == pytest.approx(2 / 3)
    assert result.macro_f1 == pytest.approx(2 / 3)
    assert result.per_class["negative"].precision == pytest.approx(2 / 3)
    assert result.per_class["negative"].recall == pytest.approx(2 / 3)
    assert result.per_class["negative"].support == 3
    assert result.per_class["positive"].precision == pytest.approx(2 / 3)
    assert result.per_class["positive"].recall == pytest.approx(2 / 3)
    assert result.per_class["positive"].support == 3
    assert result.majority_class_baseline == 0.5
    assert not result.accuracy_threshold_met
    assert not result.quality_bar_met


def test_absent_predicted_class_uses_zero_precision_and_f1():
    result = calculate_metrics(
        labels=[0, 1, 1],
        predictions=[0, 0, 0],
        n_resamples=100,
    )

    assert result.per_class["positive"].precision == 0.0
    assert result.per_class["positive"].recall == 0.0
    assert result.per_class["positive"].f1 == 0.0
    assert result.majority_class_baseline == pytest.approx(2 / 3)


def test_quality_bar_requires_both_conditions():
    predictions = [0] * 49 + [1] * 51
    labels = [0] * 50 + [1] * 50

    result = calculate_metrics(labels, predictions, n_resamples=1_000, seed=7)

    assert result.accuracy == 0.99
    assert result.accuracy_threshold_met
    assert result.accuracy_confidence_interval_95[0] > 0.5
    assert result.confidence_above_baseline
    assert result.quality_bar_met


def test_bootstrap_interval_is_seeded_and_contains_observed_accuracy():
    labels = [0, 0, 1, 1, 1]
    predictions = [0, 1, 1, 1, 0]

    first = bootstrap_accuracy_interval(labels, predictions, n_resamples=500, seed=19)
    second = bootstrap_accuracy_interval(labels, predictions, n_resamples=500, seed=19)

    assert first == second
    assert first[0] <= 0.6 <= first[1]


@pytest.mark.parametrize(
    ("labels", "predictions", "message"),
    [
        ([], [], "labels cannot be empty"),
        ([0], [0, 1], "same length"),
        ([0, 2], [0, 1], "labels must contain only 0 and 1"),
        ([0, 1], [0, 2], "predictions must contain only 0, 1, or an invalid value"),
    ],
)
def test_invalid_metric_inputs_raise(labels, predictions, message):
    with pytest.raises(ValueError, match=message):
        calculate_metrics(labels, predictions, n_resamples=10)


def test_evaluator_rejects_text_label_length_mismatch():
    evaluator = Evaluator(lambda texts: [0] * len(texts), n_resamples=10)

    with pytest.raises(ValueError, match="texts and labels"):
        evaluator.evaluate(["only one"], [0, 1])


@pytest.mark.parametrize(
    ("keyword_arguments", "message"),
    [
        ({"confidence": 0.0}, "confidence"),
        ({"confidence": 1.0}, "confidence"),
        ({"n_resamples": 0}, "n_resamples"),
    ],
)
def test_invalid_bootstrap_settings_raise(keyword_arguments, message):
    with pytest.raises(ValueError, match=message):
        bootstrap_accuracy_interval([0], [0], **keyword_arguments)


def test_bootstrap_rejects_empty_or_mismatched_inputs():
    with pytest.raises(ValueError, match="empty"):
        bootstrap_accuracy_interval([], [], n_resamples=10)
    with pytest.raises(ValueError, match="same length"):
        bootstrap_accuracy_interval([0], [0, 1], n_resamples=10)


def test_id2label_accepts_string_keys_and_case():
    validate_id2label({"0": "NEGATIVE", "1": "POSITIVE"})


def test_id2label_rejects_ambiguous_mapping():
    with pytest.raises(ValueError, match="Expected id2label"):
        validate_id2label({0: "positive", 1: "negative"})


def test_result_converts_to_nested_dictionary():
    result = calculate_metrics([0, 1], [0, 1], n_resamples=10)

    converted = result.to_dict()

    assert converted["per_class"]["positive"]["recall"] == 1.0
    assert converted["confusion_matrix"] == ((1, 0, 0), (0, 1, 0))


def test_invalid_predictions_are_counted_as_incorrect():
    result = calculate_metrics(
        [0, 1, 1],
        [INVALID_LABEL, None, 1],
        n_resamples=100,
    )

    assert result.accuracy == pytest.approx(1 / 3)
    assert result.invalid_prediction_count == 2
    assert result.invalid_prediction_rate == pytest.approx(2 / 3)
    assert result.confusion_matrix == ((0, 0, 1), (0, 1, 1))
    assert result.per_class["negative"].recall == 0.0
    assert result.per_class["positive"].recall == 0.5


@pytest.mark.parametrize(
    ("generated", "expected"),
    [
        ("positive", POSITIVE_LABEL),
        ("  NEGATIVE\n", NEGATIVE_LABEL),
        ("This review is positive.", None),
        ("positive.", None),
        ("", None),
    ],
)
def test_strict_generated_label_parser(generated, expected):
    assert parse_sentiment_label(generated) is expected


def test_sentiment_messages_use_one_shared_output_contract():
    messages = sentiment_messages("A lovely movie.")

    assert messages[-1] == {"role": "user", "content": "A lovely movie."}
    assert "exactly one label" in messages[0]["content"]
    assert "positive or negative" in messages[0]["content"]


def test_llama_predictor_uses_chat_template_and_decodes_only_new_tokens():
    model = FakeLlamaModel()
    tokenizer = FakeLlamaTokenizer()
    predictor = LlamaSST2Predictor(
        model,
        tokenizer,
        type("FakeTorch", (), {"inference_mode": nullcontext}),
        batch_size=2,
        max_new_tokens=3,
    )

    predictions = predictor(["great", "unclear"])

    assert predictions == [POSITIVE_LABEL, None]
    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token_id == tokenizer.eos_token_id
    assert tokenizer.conversations[0][-1]["content"] == "great"
    assert tokenizer.template_kwargs == {
        "add_generation_prompt": True,
        "tokenize": True,
        "padding": True,
        "return_tensors": "pt",
        "return_dict": True,
    }
    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["max_new_tokens"] == 3
    assert tokenizer.decoded_slice == (slice(None), slice(5, None))
    assert tokenizer.decode_kwargs == {"skip_special_tokens": True}


@pytest.mark.parametrize("argument", ["batch_size", "max_new_tokens"])
def test_llama_predictor_rejects_nonpositive_settings(argument):
    kwargs = {argument: 0}

    with pytest.raises(ValueError, match="positive"):
        LlamaSST2Predictor(
            FakeLlamaModel(),
            FakeLlamaTokenizer(),
            type("FakeTorch", (), {"inference_mode": nullcontext}),
            **kwargs,
        )
