import pandas as pd
from tqdm import tqdm

from src.config import DATA_PATH
from src.preprocessing import clean_resume, garbage_ratio, whitespace_ratio


def main():
    resumes = pd.read_csv(DATA_PATH)
    resume_text = resumes["Resume_str"].fillna("")

    noise_scores = pd.DataFrame(
        {
            "garbage_ratio": [garbage_ratio(t) for t in tqdm(resume_text)],
            "whitespace_ratio": [whitespace_ratio(t) for t in tqdm(resume_text)],
        }
    )
    cleaned_resumes = [clean_resume(t) for t in tqdm(resume_text)]

    print("Distribution of noise scores across all resumes:")
    print(noise_scores.describe().round(4))
    print(f"Resumes with any garbage: {(noise_scores['garbage_ratio'] > 0).sum()}")
    print("Raw vs cleaned, first 200 characters of the first resume:")
    print(repr(resume_text.iloc[0][:200]))
    print(repr(cleaned_resumes[0][:200]))
    return cleaned_resumes


if __name__ == "__main__":  # pragma: no cover
    main()
