import json
import re


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_text(obj):
    """Recursively extract all strings from a JSON object."""
    texts = []

    if isinstance(obj, str):
        texts.append(obj)
    elif isinstance(obj, list):
        for item in obj:
            texts.extend(extract_text(item))
    elif isinstance(obj, dict):
        for value in obj.values():
            texts.extend(extract_text(value))

    return texts


def preprocess_text(text):
    """Split text into words and punctuation, removing whitespace-only items."""
    tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    return [item.strip() for item in tokens if item.strip()]


def process_json(file_path):
    """Load JSON, extract its text, and return both raw text and preprocessed tokens."""
    data = load_json(file_path)
    all_text = " ".join(extract_text(data))
    preprocessed = preprocess_text(all_text)
    return all_text, preprocessed
