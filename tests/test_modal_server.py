import pytest

from src import modal_server


def test_kv_cache_counts_keys_and_values_for_every_layer_head_and_token():
    # 2 (K and V) * 28 layers * 8 kv heads * 128 head dim * 2 bytes * 1000 tokens
    mib = modal_server.kv_cache_mib(layers=28, kv_heads=8, head_dim=128, tokens=1000)
    assert mib == pytest.approx(109.375)


# ---- the Modal class and entrypoints, without a GPU ----
# These reach inside Modal: _get_user_cls() is the plain class behind @app.cls and _get_raw_f() the plain
# function behind @modal.method(). Both are Modal-internal, so a Modal upgrade may need these helpers updated.

import json
import time
from types import SimpleNamespace

from src import config, methods
from src.html import to_html
from src.schema import WebPage

GOOD_JSON = '{"title": "t", "body": [{"type": "heading", "level": 1, "text": "h"}]}'
GOOD_HTML = "<html><head><title>t</title></head><body><h1>h</h1></body></html>"


class FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        assert add_generation_prompt is True and tokenize is False
        assert [m["role"] for m in messages] == ["user"]
        return f"<chat>{messages[0]['content']}</chat>"


def generator():
    """A Generator with load() skipped, since load() needs the GPU and the real model."""
    cls = modal_server.Generator._get_user_cls()
    obj = cls.__new__(cls)
    obj.tokenizer, obj.model, obj.gpu = FakeTokenizer(), None, "Fake GPU"
    obj.container_id, obj.started, obj.calls = "c1", time.time(), {}
    return obj


def test_meta_counts_calls_for_each_method_separately():
    obj = generator()
    first, second = obj.meta("guidance"), obj.meta("guidance")
    other = obj.meta("coercion")
    assert (first["calls_before"], second["calls_before"], other["calls_before"]) == (
        0,
        1,
        0,
    )
    assert first["container_id"] == "c1" and first["container_age_s"] >= 0


def test_warm_state_reports_the_counts_as_a_copy():
    obj = generator()
    obj.meta("guidance")
    state = obj.warm_state()
    assert state == {"guidance": 1}
    state["guidance"] = 99
    assert obj.calls == {"guidance": 1}


def test_chat_text_formats_one_user_message_with_the_generation_prompt():
    assert generator().chat_text("hello") == "<chat>hello</chat>"


def test_method_a_starts_from_the_shared_chat_prompt_and_keeps_html_only_when_valid(
    monkeypatch,
):
    seen = []

    def fake_run_guidance(model, tokenizer, chat):
        seen.append(chat)
        return {"text": GOOD_HTML, "new_tokens": 12, "hit_cap": False}

    monkeypatch.setattr(methods, "run_guidance", fake_run_guidance)
    obj = generator()
    record = obj.run_guidance("a bakery")
    assert seen == [
        obj.chat_text(config.BASE_PROMPT_TEMPLATE.format(request="a bakery"))
    ]
    assert (record["method"], record["bucket"], record["html"]) == (
        "guidance",
        "valid",
        GOOD_HTML,
    )
    assert (record["gpu"], record["model"], record["new_tokens"]) == (
        "Fake GPU",
        config.MODEL_NAME,
        12,
    )
    assert record["calls_before"] == 0 and record["seconds"] >= 0


def test_method_a_cut_off_by_the_cap_is_truncated_and_has_no_html(monkeypatch):
    monkeypatch.setattr(
        methods,
        "run_guidance",
        lambda m, t, c: {"text": "<html><head>", "new_tokens": 500, "hit_cap": True},
    )
    record = generator().run_guidance("a bakery")
    assert (record["bucket"], record["html"], record["hit_cap"]) == (
        "truncated",
        None,
        True,
    )


def test_method_b_sends_the_schema_prompt_and_assembles_html_from_valid_json():
    obj, prompts = generator(), []
    obj.complete = lambda prompt: (prompts.append(prompt), GOOD_JSON, 30, False)[1:]
    record = obj.run_coercion("a bakery")
    assert prompts == [methods.build_prompt("a bakery")]
    assert (record["method"], record["bucket"]) == ("coercion", "valid")
    assert record["html"] == to_html(WebPage.model_validate_json(GOOD_JSON))
    assert record["text"] == GOOD_JSON


def test_method_b_markdown_fence_is_malformed_and_has_no_html():
    obj = generator()
    obj.complete = lambda prompt: (f"```json\n{GOOD_JSON}\n```", 30, False)
    record = obj.run_coercion("a bakery")
    assert (record["bucket"], record["html"]) == ("malformed", None)


class FakeVolume:
    """Stands in for the Modal results volume and remembers the order of reload and commit."""

    def __init__(self):
        self.events = []

    def reload(self):
        self.events.append("reload")

    def commit(self):
        self.events.append("commit")


def full_record(method):
    return {
        "method": method, "bucket": "valid", "reason": "", "html": "<html></html>", "text": "x",
        "new_tokens": 1, "hit_cap": False, "seconds": 1.0,
    }  # fmt: skip


def run_all_on_a_fake_container(tmp_path, monkeypatch):
    """The whole loop runs inside the container, so it is driven here with fake calls and a fake volume."""
    volume = FakeVolume()
    monkeypatch.setattr(modal_server, "results_volume", volume)
    monkeypatch.setattr(modal_server, "RESULTS_DIR", str(tmp_path))
    obj = generator()
    obj.probe = lambda: {"gpu": "Fake GPU"}
    obj.calls_made = []
    obj.run_guidance = lambda request: (
        obj.calls_made.append(request),
        full_record("guidance"),
    )[1]
    obj.run_coercion = lambda request: (
        obj.calls_made.append(request),
        full_record("coercion"),
    )[1]
    run_all = modal_server.Generator._get_user_cls().run_all._get_raw_f()
    return volume, obj, lambda name, count, **kw: run_all(obj, name, count, **kw)


def test_run_all_runs_every_prompt_through_both_methods_and_saves_to_the_volume(
    tmp_path, monkeypatch
):
    volume, _, run_all = run_all_on_a_fake_container(tmp_path, monkeypatch)
    summary = run_all("t1", 3)
    data = json.loads((tmp_path / "benchmark_t1.json").read_text())
    assert len(data["records"]) == 6 and data["probe"]["gpu"] == "Fake GPU"
    assert len(data["warmups"]) == 2  # a fresh container warms each method up once
    assert summary == {"path": str(tmp_path / "benchmark_t1.json"), "records": 6}
    assert volume.events == ["reload", "commit"]


def test_run_all_never_overwrites_a_run_unless_asked(tmp_path, monkeypatch):
    _, obj, run_all = run_all_on_a_fake_container(tmp_path, monkeypatch)
    run_all("t1", 1)
    calls = len(obj.calls_made)
    with pytest.raises(FileExistsError):
        run_all("t1", 1)
    assert len(obj.calls_made) == calls  # refused before spending any GPU time
    run_all("t1", 2, overwrite=True)
    assert len(json.loads((tmp_path / "benchmark_t1.json").read_text())["records"]) == 4


class FakeGenerator:
    def __init__(self):
        self.calls = []
        self.run_all = SimpleNamespace(remote=self.remote)

    def remote(self, name, count, overwrite=False):
        self.calls.append((name, count, overwrite))
        return {"path": f"/results/benchmark_{name}.json", "records": 2 * count}


def test_run_can_be_limited_to_the_first_few_prompts(monkeypatch):
    fake = FakeGenerator()
    monkeypatch.setattr(modal_server, "Generator", lambda: fake)
    modal_server.run.info.raw_f("diag1", 3)
    assert fake.calls == [("diag1", 3, False)]


def test_run_asks_for_every_prompt_and_forbids_overwriting(monkeypatch):
    fake = FakeGenerator()
    monkeypatch.setattr(modal_server, "Generator", lambda: fake)
    modal_server.run.info.raw_f("run1")
    assert fake.calls == [("run1", len(config.PROMPTS), False)]


def test_smoke_asks_for_two_prompts_and_may_overwrite_its_own_file(monkeypatch):
    fake = FakeGenerator()
    monkeypatch.setattr(modal_server, "Generator", lambda: fake)
    modal_server.smoke.info.raw_f()
    assert fake.calls == [("smoke", 2, True)]
