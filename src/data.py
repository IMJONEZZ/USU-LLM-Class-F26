import re
import unicodedata

from src import config

_FINAL_LABEL_WORDS = config.FINAL_ANSWER_LABEL.rstrip(":").split()
_FINAL_LABEL_RE = re.compile(
    r"^[ \t]*" + r"[ \t]+".join(_FINAL_LABEL_WORDS) + r"[ \t]*:",
    re.IGNORECASE | re.MULTILINE,
)
_MARKERS = [config.INSTRUCTION_MARKER.strip(), config.RESPONSE_MARKER.strip()]
_APOSTROPHES = "'’‘"


def _reject_markers(text, what):
    for marker in _MARKERS:
        if marker in text:
            raise ValueError(f"{what} contains template marker {marker!r}")


def format_prompt(question):
    _reject_markers(question, "question")
    question = question.strip()
    if not question:
        raise ValueError("question is empty")
    return f"{config.INSTRUCTION_MARKER}{question}\n\n{config.RESPONSE_MARKER}"


def _check_chain(chain):
    _reject_markers(chain, "chain")
    lines = chain.split("\n")
    positions = []
    for label in config.SECTION_LABELS:
        found = [i for i, line in enumerate(lines) if line.lstrip().startswith(label)]
        if len(found) != 1:
            raise ValueError(f"expected exactly one {label!r} line, found {len(found)}")
        positions.append(found[0])
    if positions != sorted(positions):
        raise ValueError("chain sections are out of order")
    final_line = lines[positions[-1]].lstrip()
    if not final_line[len(config.FINAL_ANSWER_LABEL) :].strip():
        raise ValueError("final answer is empty")


def format_example(question, chain, eos_token):
    chain = chain.strip()
    if chain.endswith(eos_token):
        chain = chain[: -len(eos_token)].strip()
    _check_chain(chain)
    return format_prompt(question) + chain + eos_token


def format_fewshot(examples, target_question, max_chars=None):
    target_norm = normalize(target_question)
    pieces = []
    for example in examples:
        if normalize(example["question"]) == target_norm:
            raise ValueError("a few-shot example is identical to the target question")
        chain = example["chain"].strip()
        _check_chain(chain)
        pieces.append(format_prompt(example["question"]) + chain)
    pieces.append(format_prompt(target_question))
    out = "\n\n".join(pieces)
    if max_chars is not None and len(out) > max_chars:
        raise ValueError(f"few-shot prompt is {len(out)} chars, limit is {max_chars}")
    return out


def extract_final_answer(text):
    cut = len(text)
    for marker in _MARKERS:
        idx = text.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    text = text[:cut]
    match = _FINAL_LABEL_RE.search(text)
    if match is None:
        return None
    parts = []
    for line in text[match.end() :].split("\n"):
        stripped = line.strip()
        if not stripped:
            break
        if parts and (
            _FINAL_LABEL_RE.match(line)
            or any(stripped.startswith(label) for label in config.SECTION_LABELS)
        ):
            break
        parts.append(stripped)
    answer = " ".join(parts)
    return answer or None


def normalize(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).casefold().replace("−", "-")
    chars = []
    for i, ch in enumerate(text):
        if ch in _APOSTROPHES:
            continue
        is_minus = (
            ch == "-"
            and text[i + 1 : i + 2].isdigit()
            and not text[i - 1 : i].isalnum()
        )
        if unicodedata.category(ch).startswith("P") and ch != "." and not is_minus:
            chars.append(" ")
        else:
            chars.append(ch)
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", "".join(chars))
    return " ".join(text.split())


def is_correct(predicted, key, alternatives=()):
    accepted = {normalize(k) for k in (key, *alternatives)} - {""}
    if not normalize(key):
        raise ValueError("answer key is empty")
    predicted_norm = normalize(predicted)
    return predicted_norm != "" and predicted_norm in accepted
