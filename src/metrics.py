import re

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_ID


def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_ID)


def normalize_label(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def is_valid_category(generated, categories):
    return normalize_label(generated) in {normalize_label(c) for c in categories}


def semantic_similarity(generated, expected, model):
    generated_vector, expected_vector = model.encode([generated, expected])
    return float(
        np.dot(generated_vector, expected_vector)
        / (np.linalg.norm(generated_vector) * np.linalg.norm(expected_vector))
    )
