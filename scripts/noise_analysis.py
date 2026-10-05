import string

import pandas as pd
from tqdm import tqdm

from src.config import DATA_PATH
from src.preprocessing import normalize_text

ALLOWED_CHARS = set(
    string.ascii_letters + string.digits + string.whitespace + string.punctuation
)


def garbage_ratio(text):
    normalized = normalize_text(text)
    if not normalized:
        return 0.0
    return sum(ch not in ALLOWED_CHARS for ch in normalized) / len(normalized)


resumes = pd.read_csv(DATA_PATH)
resume_text = resumes["Resume_str"].fillna("")
garbage_ratios = pd.Series([garbage_ratio(t) for t in tqdm(resume_text)])

# and then we ask whether garbage_ratio separates resumes at all
print("Resumes with garbage_ratio > 0, out of all resumes:")
print((garbage_ratios > 0).sum(), "of", len(garbage_ratios))
print("Distribution of garbage_ratio across all resumes:")
print(garbage_ratios.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).round(4))
