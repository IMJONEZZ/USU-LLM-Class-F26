import json


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


def process_json(file_path):
    """Load JSON, extract its text, and return both raw text and preprocessed tokens."""
    data = load_json(file_path)
    all_text = " ".join(extract_text(data))
    return all_text
