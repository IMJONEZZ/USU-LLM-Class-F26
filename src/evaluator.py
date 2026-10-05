from tqdm import tqdm

from src.config import SEED
from src.metrics import is_valid_category, normalize_label, semantic_similarity
from src.prompts import format_prompt


def load_eval_subset(test_resumes, n=100, seed=SEED):
    return test_resumes.sample(frac=1, random_state=seed).head(n)


def majority_category(train_resumes):
    return train_resumes["Category"].mode()[0]


def evaluate_model(
    generate_fn, eval_resumes, categories, embedding_model, batch_size=8
):
    prompts = [
        format_prompt(resume_text, categories)
        for resume_text in eval_resumes["Resume_str"]
    ]
    generated_labels = []
    for start in tqdm(range(0, len(prompts), batch_size)):
        generated_labels.extend(generate_fn(prompts[start : start + batch_size]))

    expected_labels = list(eval_resumes["Category"])
    similarities = [
        semantic_similarity(generated, expected, embedding_model)
        for generated, expected in zip(generated_labels, expected_labels)
    ]
    valid_flags = [is_valid_category(g, categories) for g in generated_labels]
    exact_flags = [
        normalize_label(generated) == normalize_label(expected)
        for generated, expected in zip(generated_labels, expected_labels)
    ]
    return {
        "n": len(expected_labels),
        "mean_semantic_similarity": sum(similarities) / len(similarities),
        "valid_category_rate": sum(valid_flags) / len(valid_flags),
        "exact_match_rate": sum(exact_flags) / len(exact_flags),
    }
