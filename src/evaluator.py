"""Evaluates a BERT model on paraphrase detection using GLUE MRPC.

The model is textattack/bert-base-uncased-MRPC and the data is the MRPC
validation split. Accuracy on its own is misleading on this dataset, so
everything here reports the majority baseline next to it.

Run it from the repo root:

    uv run python -m src.evaluator
"""

from collections import Counter

DEFAULT_MODEL = "textattack/bert-base-uncased-MRPC"
DEFAULT_DATASET = "nyu-mll/glue"
DEFAULT_CONFIG = "mrpc"
DEFAULT_SPLIT = "validation"

# There's no id2label in this model's config so the pipeline just hands back
# LABEL_0 and LABEL_1. I checked it against the gold labels instead of guessing,
# the mapping below gets 0.85 on 40 rows and the flipped one gets 0.15.
LABEL_MAPPING = {"LABEL_0": 0, "LABEL_1": 1}


def load_mrpc(split=DEFAULT_SPLIT, limit=None):
    """Returns the sentence pairs and the gold labels for a MRPC split."""
    # datasets is slow to import and most of the tests never touch it, so this
    # stays down here instead of at the top of the file.
    from datasets import load_dataset

    rows = load_dataset(DEFAULT_DATASET, DEFAULT_CONFIG, split=split)
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))

    pairs = list(zip(rows["sentence1"], rows["sentence2"]))
    return pairs, list(rows["label"])


def majority_baseline(labels):
    """What a model scores if it ignores the input and always guesses the
    most common label.

    MRPC is lopsided. 279 of the 408 validation pairs are labeled equivalent,
    so answering "equivalent" every time already gets 68.4% without the model
    knowing anything. Accuracy has to be read against that and not against 0.
    """
    if not labels:
        return 0.0
    counts = Counter(labels)
    return counts.most_common(1)[0][1] / len(labels)


def load_glue_metric():
    # This downloads a small builder script the first time, which is why
    # evaluate_predictor lets you pass a metric in instead.
    import evaluate

    return evaluate.load("glue", DEFAULT_CONFIG)


def build_predictor(
    model_name=DEFAULT_MODEL, label_mapping=None, batch_size=16, classifier=None
):
    """Wraps the model in a function that turns sentence pairs into 0s and 1s.

    Pass a classifier in and it gets used as is, which is how the tests avoid
    downloading a few hundred megabytes of weights. Leave it out and the real
    pipeline gets built. Either way the thing handed back takes a list of
    (sentence1, sentence2) and gives back a list of ints.
    """
    mapping = LABEL_MAPPING if label_mapping is None else label_mapping

    # Not covered by the tests on purpose. Exercising this branch means
    # actually downloading the model, which is the thing I was avoiding.
    if classifier is None:  # pragma: no cover
        from transformers import pipeline

        classifier = pipeline("text-classification", model=model_name)

    def predict(pairs):
        if not pairs:
            return []
        inputs = [{"text": first, "text_pair": second} for first, second in pairs]
        outputs = classifier(inputs, batch_size=batch_size)
        # If the label isn't in the mapping I'd rather blow up here than
        # quietly score against something I don't understand.
        return [mapping[output["label"]] for output in outputs]

    return predict


def evaluate_predictor(predictor, pairs, labels, metric=None):
    """Scores a predictor and returns the numbers needed to read the result.

    metric is anything with a compute(predictions=, references=) that gives
    back accuracy and f1, which is the shape of the GLUE metric.
    """
    if len(pairs) != len(labels):
        raise ValueError(f"got {len(pairs)} sentence pairs but {len(labels)} labels")

    predictions = predictor(pairs)
    if len(predictions) != len(labels):
        raise ValueError(
            f"predictor returned {len(predictions)} predictions "
            f"for {len(labels)} inputs"
        )

    scorer = load_glue_metric() if metric is None else metric
    scores = scorer.compute(predictions=predictions, references=labels)

    baseline = majority_baseline(labels)
    return {
        "n": len(labels),
        "accuracy": scores["accuracy"],
        "f1": scores["f1"],
        "majority_baseline": baseline,
        # This is the number I actually care about and it's the one plain
        # accuracy hides.
        "lift_over_baseline": scores["accuracy"] - baseline,
        "predictions": predictions,
    }


def main(limit=None):
    pairs, labels = load_mrpc(limit=limit)
    predictor = build_predictor()
    results = evaluate_predictor(predictor, pairs, labels)

    print(f"model              {DEFAULT_MODEL}")
    print(f"dataset            {DEFAULT_DATASET} ({DEFAULT_CONFIG}, {DEFAULT_SPLIT})")
    print(f"examples           {results['n']}")
    print(f"accuracy           {results['accuracy']:.4f}")
    print(f"f1                 {results['f1']:.4f}")
    print(f"majority baseline  {results['majority_baseline']:.4f}")
    print(f"lift over baseline {results['lift_over_baseline']:+.4f}")
    return results


if __name__ == "__main__":  # pragma: no cover
    main()
