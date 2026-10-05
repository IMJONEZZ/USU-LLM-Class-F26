import pandas as pd

from src.config import DATA_PATH
from src.preprocessing import sample_stratified


def main():
    resumes = pd.read_csv(DATA_PATH)
    sampled_resumes = sample_stratified(resumes)

    print(f"Sampled {len(sampled_resumes)} of {len(resumes)} resumes")
    print("Category share: all resumes vs sampled resumes:")
    print(
        pd.DataFrame(
            {
                "all_share": resumes["Category"].value_counts(normalize=True),
                "sampled_share": sampled_resumes["Category"].value_counts(
                    normalize=True
                ),
            }
        ).round(3)
    )
    return sampled_resumes


if __name__ == "__main__":  # pragma: no cover
    main()
