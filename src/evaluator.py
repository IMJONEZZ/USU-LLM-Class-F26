from datasets import load_dataset  # pragma: no cover
from evaluate import evaluator


# Wrapping logic in function allows you to evaluate subsets of the dataset (split arg)
def run_evaluation(
    model_name: str = "csarron/bert-base-uncased-squad-v1", split: str = "validation"
) -> dict:
    """Evaluate a SQuAD-fine-tuned BERT model on AdversarialQA."""
    # Load the AdversarialQA dataset from Hugging Face Datasets
    data = load_dataset("UCLNLP/adversarial_qa", "adversarialQA", split=split)

    qa_evaluator = evaluator("question-answering")

    # squad metric returns the average exact match and F1 scores
    return qa_evaluator.compute(model_or_pipeline=model_name, data=data, metric="squad")


if __name__ == "__main__":  # pragma: no cover
    print(run_evaluation())
