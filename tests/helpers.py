import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

from tokenizers import Tokenizer, models, pre_tokenizers, processors

from src import config, data, teacher, train

ROOT = Path(__file__).resolve().parents[1]


# ---- generate_chain ----
ROW = {
    "id": "x-1",
    "question": "How many sheep are left?",
    "answer": "9",
    "accepted_alternatives": ["nine"],
    "category": "lateral_thinking",
    "source": "brainteaser",
}


def _teacher_chain(final="9", filler=""):
    return (
        "Restate: A farmer has 17 sheep; all but 9 run away.\n"
        "Check: This is a wording trap, not subtraction.\n"
        f"Solve: All but 9 means 9 remain. {filler}\n"
        "Verify: 17 - 8 = 9, so 9 stay.\n"
        f"Final answer: {final}"
    )


def _without_line(chain, label):
    return "\n".join(l for l in chain.split("\n") if not l.startswith(label))


# ---- build_training_data: one permanent chain store, nothing is ever redone ----
def _train_rows(n=4):
    return [
        {
            "id": f"r-{i}",
            "question": f"Question number {i}?",
            "answer": "9",
            "accepted_alternatives": ["nine"],
            "category": config.CATEGORIES[i % len(config.CATEGORIES)],
            "source": "brainteaser",
        }
        for i in range(n)
    ]


def _fake_generate_chain(monkeypatch, fail_ids=(), crash_on=None, crash_with=None):
    seen = []

    def fake(row):
        seen.append(row)
        if row["id"] == crash_on:
            raise crash_with or RuntimeError("boom")
        if row["id"] in fail_ids:
            return None
        return {**row, "chain": "CHAIN-" + row["id"]}

    monkeypatch.setattr(teacher, "generate_chain", fake)
    return seen


def _paths(tmp_path):
    return tmp_path / "train.jsonl", tmp_path / "chains.jsonl"


# ---- scripts/pipeline.py (no network: these paths exit before any API call) ----
PIPELINE = str(ROOT / "scripts" / "pipeline.py")


def _run_pipeline(*args):
    return subprocess.run(
        [sys.executable, PIPELINE, *args], capture_output=True, text=True, check=False
    )


# ---- M4: test set, leakage and hash ----
def _trow(i=1, **overrides):
    return {
        "id": f"t-{i}",
        "question": f"Hand written question number {i}?",
        "answer": "9",
        "accepted_alternatives": ["nine"],
        "category": "lateral_thinking",
        "source": "hand_written",
        "chain": _teacher_chain("9", f"variant {i}"),
        **overrides,
    }


def _train_row(i=1, **overrides):
    return {
        "id": f"tr-{i}",
        "question": f"Public training question {i}?",
        "answer": "a",
        "accepted_alternatives": [],
        "category": "misconception",
        "source": "truthfulqa",
        "chain": _teacher_chain("a", f"train {i}"),
        **overrides,
    }


# ---- load_test_set ----
def _write_test_set(tmp_path, rows):
    path = tmp_path / "test_set.jsonl"
    data.write_jsonl(path, rows)
    return path


# ---- scripts/pipeline.py: test-data and validate ----
def _write_all(tmp_path, test_rows, train_rows, eval_rows=None):
    test_path = tmp_path / "test_set.jsonl"
    train_path = tmp_path / "train.jsonl"
    data.write_jsonl(test_path, test_rows)
    data.write_jsonl(train_path, train_rows)
    if eval_rows is not None:
        data.write_jsonl(tmp_path / "eval.jsonl", eval_rows)
    return test_path, train_path, tmp_path / "eval.jsonl", tmp_path / "hash.jsonl"


# ---- M5: make_variants (code-side filters on the model's rewordings) ----
LONG_Q = "If it takes 1 hour for 60 people to play an opera, how many hours will it take 600 people?"


def _source_row(**overrides):
    return _trow(
        1,
        question=LONG_Q,
        answer="1",
        accepted_alternatives=["1 hour"],
        chain=_teacher_chain("1"),
        **overrides,
    )


def _reworded(word):
    return LONG_Q.replace("play", word)


# ---- M5: build_eval_data (permanent store; the paraphraser and judge are faked) ----
QUESTIONS = [
    "Why does the sky look blue during the day?",
    "How do planes stay in the air?",
    "What makes bread rise in the oven?",
]


def _qrow(i, **overrides):
    return _trow(i, question=QUESTIONS[i - 1], **overrides)


def _cands(question, n, start=0):
    return [f"{question[:-1]} indeed{'x' * (start + k)}?" for k in range(n)]


class _Spend:
    def __init__(self, verdicts=None, errors=()):
        self.verdicts = verdicts or {}
        self.errors = set(errors)
        self.asked = []
        self.judged = []
        self.calls_to_paraphrase = 0

    def paraphrase(self, question, n):
        self.asked.append((question, n))
        return _cands(question, n)

    def judge(self, source_row, variant_question):
        self.judged.append(variant_question)
        if variant_question in self.errors:
            return {"keep": False, "reason": "judge_error"}
        return self.verdicts.get(variant_question, {"keep": True, "reason": "kept"})


def _eval_paths(tmp_path):
    return (
        tmp_path / "eval.jsonl",
        tmp_path / "eval_review.jsonl",
        tmp_path / "eval_store.jsonl",
    )


def _build(tmp_path, monkeypatch, rows, spend=None, per_question=2, paraphrase=None):
    spend = spend or _Spend()
    monkeypatch.setattr(teacher, "judge_variant", spend.judge)
    out, review, store = _eval_paths(tmp_path)
    summary = teacher.build_eval_data(
        rows,
        out,
        review,
        store,
        per_question=per_question,
        paraphrase=paraphrase or spend.paraphrase,
    )
    return summary, spend


# ---- M6: token audit ----
def _tiny_tokenizer(with_bos=True):
    tok = Tokenizer(models.WordLevel({"<unk>": 0}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(["<unk>", "<bos>", config.EOS_TOKEN])
    if with_bos:
        tok.post_processor = processors.TemplateProcessing(
            single="<bos> $A", special_tokens=[("<bos>", tok.token_to_id("<bos>"))]
        )
    return tok


# ---- M6: validate --tokens ----
def _tiny_tokenizer_file(tmp_path):
    path = tmp_path / "tok.json"
    _tiny_tokenizer().save(str(path))
    return path


def _validate_tokens(tmp_path, *extra, eval_rows=()):
    paths = _write_all(
        tmp_path, [_trow(1)], [_train_row(1), _train_row(2)], list(eval_rows)
    )
    test_path, train_path, eval_path, hash_path = paths
    return _run_pipeline(
        "validate",
        "--test-set", str(test_path),
        "--train", str(train_path),
        "--eval", str(eval_path),
        "--hash", str(hash_path),
        "--tokens",
        "--tokenizer", str(_tiny_tokenizer_file(tmp_path)),
        "--token-audit", str(tmp_path / "audit.json"),
        *extra,
    )  # fmt: skip


# ---- M7: grade ----
def _grade_row(**overrides):
    return _trow(1, answer="9", accepted_alternatives=["nine"], **overrides)


# ---- M7: weak categories ----
def _acc(**values):
    return values


# ---- M8: model loading and adapters (unsloth is faked) ----
def _fake_unsloth(monkeypatch):
    module = ModuleType("unsloth")
    fast = MagicMock()
    fast.from_pretrained.return_value = ("base_model", "tokenizer")
    fast.get_peft_model.return_value = "peft_model"
    module.FastLanguageModel = fast
    monkeypatch.setitem(sys.modules, "unsloth", module)
    return fast


# ---- M8: the per-epoch evaluation ----
def _evaluator(outputs=None, raise_in_generate=False):
    rows = [
        _trow(1, answer="9", category="lateral_thinking"),
        _trow(2, answer="9", category="misconception"),
    ]
    outputs = outputs or [_teacher_chain("9"), _teacher_chain("12")]

    def generate(prompts):
        if raise_in_generate:
            raise RuntimeError("generation failed")
        return list(outputs)

    return train.EpochEvaluator(rows, generate)


# ---- both tokenizer APIs: `tokenizers` (encode gives .ids) and transformers (encode gives a list) ----
class _HFStyleTokenizer:
    """Behaves like the Hugging Face tokenizer Unsloth returns inside the container."""

    def __init__(self, inner):
        self._inner = inner

    def encode(self, text, add_special_tokens=True):
        return self._inner.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids, skip_special_tokens=True):
        return self._inner.decode(ids, skip_special_tokens=skip_special_tokens)


# ---- M9: build_round_dataset ----
def _pool_row(category, i):
    return {
        "id": f"{category}-{i}",
        "question": f"Round pool question about {category} number {i} here?",
        "answer": "a",
        "accepted_alternatives": [],
        "category": category,
        "source": "truthfulqa",
        "chain": _teacher_chain("a", f"{category} {i}"),
    }


def _round_pool(per_category=5):
    return [_pool_row(c, i) for c in config.CATEGORIES for i in range(per_category)]


def _extra(i, category="pun_ambiguity", **overrides):
    return {
        "id": f"extra-{i}",
        "question": f"A brand new written question about topic {i} for the model?",
        "answer": "a",
        "accepted_alternatives": [],
        "category": category,
        "source": "user_written",
        "chain": _teacher_chain("a", f"extra {i}"),
        **overrides,
    }


def _round(**overrides):
    args = {
        "train_rows": _round_pool(),
        "weak_categories": ["pun_ambiguity"],
        "extra_rows": [_extra(1), _extra(2), _extra(3), _extra(4)],
        "replay_fraction": 1.0,
        "seed": 0,
    }
    return data.build_round_dataset(**{**args, **overrides})
