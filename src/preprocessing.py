import re
import string
import unicodedata

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
ALLOWED_CHARS = set(
    string.ascii_letters + string.digits + string.whitespace + string.punctuation
)

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


def garbage_ratio(text):
    normalized = normalize_text(text)
    if not normalized:
        return 0.0
    return sum(ch not in ALLOWED_CHARS for ch in normalized) / len(normalized)


def whitespace_ratio(text):
    normalized = normalize_text(text)
    if not normalized:
        return 0.0
    return sum(ch.isspace() for ch in normalized) / len(normalized)


def clean_resume(text):
    normalized = normalize_text(text)
    without_links = strip_links_and_handles(normalized)
    return redact_pii(without_links).strip()
