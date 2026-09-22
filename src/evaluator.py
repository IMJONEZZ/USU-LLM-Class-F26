from datasets import load_dataset  # pragma: no cover
from evaluate import evaluator

data = load_dataset("UCLNLP/adversarial_qa", "adversarialQA", split="validation")

qa_evaluator = evaluator("question-answering")

results = qa_evaluator.compute(
    model_or_pipeline="csarron/bert-base-uncased-squad-v1", data=data, metric="squad"
)

print(results)
