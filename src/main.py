import pandas as pd

from src.config import DATA_PATH
from src.preprocessing import sample_stratified, split_stratified


def main():
    resumes = pd.read_csv(DATA_PATH)
    sampled_resumes = sample_stratified(resumes)
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
