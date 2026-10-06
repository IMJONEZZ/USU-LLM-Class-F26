import pandas as pd

from src.config import DATA_PATH
from src.preprocessing import (
    clean_resume_column,
    sample_stratified,
    split_stratified,
)


def main():
    resumes = pd.read_csv(DATA_PATH)
    cleaned_resumes = clean_resume_column(resumes)
    sampled_resumes = sample_stratified(cleaned_resumes)
    train, validation, test = split_stratified(sampled_resumes)

    print(f"Sampled {len(sampled_resumes)} of {len(resumes)} resumes")
    print(
        f"Split sizes: train {len(train)}, "
        f"validation {len(validation)}, test {len(test)}"
    )
    print("Resumes per category in each split:")
    print(
        pd.DataFrame(
            {
                "train": train["Category"].value_counts(),
                "validation": validation["Category"].value_counts(),
                "test": test["Category"].value_counts(),
            }
        )
        .fillna(0)
        .astype(int)
    )
    return train, validation, test


if __name__ == "__main__":  # pragma: no cover
    main()
