import nltk
import pytest


@pytest.fixture(scope="session", autouse=True)
def sentence_data():
    # Google's sentence and word counters need this data; CI starts without it and workflow files cannot be edited from here
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
