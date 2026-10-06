import re
import unicodedata

import pandas as pd

from src.config import N_RESUMES, SEED, SPLIT_FRACTIONS

NON_ASCII_MAP = {
    "\xa0": " ",
    "\u200b": "",
    "\xad": "",
    "\u2028": "\n",
    "\uff0d": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u2010": "-",
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": ",",
    "\u2026": "...",
    "\u2122": "TM",
    "\u2022": "-",
    "\xb7": "-",
    "\u25cf": "-",
    "\u25e6": "-",
    "\u27a2": "-",
    "\u2756": "-",
    "\u25c6": "-",
    "\u25aa": "-",
    "\u274f": "-",
    "\uf0a7": "-",
}

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_PATTERN = re.compile(r"\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}")
HANDLE_PATTERN = re.compile(r"(?<!\w)@\w+")


def normalize_text(text):
    mapped = text.translate(str.maketrans(NON_ASCII_MAP))
    decomposed = unicodedata.normalize("NFKD", mapped)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def strip_links_and_handles(text):
    without_urls = URL_PATTERN.sub("", text)
    without_emails = EMAIL_PATTERN.sub("[EMAIL]", without_urls)
    return HANDLE_PATTERN.sub("", without_emails)


def redact_pii(text):
    return PHONE_PATTERN.sub("[PHONE]", text)


def clean_resume(text):
    normalized = normalize_text(text)
    without_links = strip_links_and_handles(normalized)
    return redact_pii(without_links).strip()


def clean_resume_column(resumes):
    return resumes.assign(Resume_str=resumes["Resume_str"].fillna("").map(clean_resume))


def allocate_proportional_quotas(available_per_category, n_total):
    exact_quotas = available_per_category / available_per_category.sum() * n_total
    quotas = exact_quotas.astype(int)
    shortfall = n_total - quotas.sum()
    largest_remainders = (exact_quotas - quotas).sort_values(
        ascending=False, kind="stable"
    )
    quotas.loc[largest_remainders.index[:shortfall]] += 1
    return quotas


def sample_stratified(resumes, n_total=N_RESUMES, seed=SEED):
    shuffled_resumes = resumes.sample(frac=1, random_state=seed)
    quotas = allocate_proportional_quotas(
        shuffled_resumes["Category"].value_counts(), n_total
    )
    sampled_by_category = [
        shuffled_resumes[shuffled_resumes["Category"] == category].head(quota)
        for category, quota in quotas.items()
    ]
    return pd.concat(sampled_by_category)


def split_stratified(
    resumes, stratify_by="Category", fractions=SPLIT_FRACTIONS, seed=SEED
):
    shuffled_resumes = resumes.sample(frac=1, random_state=seed)
    parts_per_split = [[] for _ in fractions]
    for _, group in shuffled_resumes.groupby(stratify_by):
        quotas = allocate_proportional_quotas(pd.Series(fractions), len(group))
        start = 0
        for split_index, quota in enumerate(quotas):
            parts_per_split[split_index].append(group.iloc[start : start + quota])
            start += quota
    return tuple(pd.concat(parts) for parts in parts_per_split)


def make_splits(resumes):
    cleaned_resumes = clean_resume_column(resumes)
    sampled_resumes = sample_stratified(cleaned_resumes)
    return split_stratified(sampled_resumes)
