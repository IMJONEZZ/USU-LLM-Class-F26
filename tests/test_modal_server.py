import pytest

from src import modal_server

EOS_IDS = {128001, 128009}


def test_kv_cache_counts_keys_and_values_for_every_layer_head_and_token():
    # 2 (K and V) * 28 layers * 8 kv heads * 128 head dim * 2 bytes * 1000 tokens
    mib = modal_server.kv_cache_mib(layers=28, kv_heads=8, head_dim=128, tokens=1000)
    assert mib == pytest.approx(109.375)


@pytest.mark.parametrize(
    ("token_ids", "expected"),
    [
        ([1, 2, 3, 4], True),  # hit the cap without ending
        (
            [1, 2, 3, 128009],
            False,
        ),  # hit the cap but the last token is end-of-turn: it finished
        ([1, 2, 3], False),  # stopped below the cap
        ([], False),
    ],
)
def test_is_truncated_only_when_the_cap_is_hit_without_an_end_token(
    token_ids, expected
):
    assert modal_server.is_truncated(token_ids, EOS_IDS, cap=4) is expected


# ---- the Modal class and entrypoints, without a GPU ----
# These reach inside Modal: _get_user_cls() is the plain class behind @app.cls and _get_raw_f() the plain
# function behind @modal.method(). Both are Modal-internal, so a Modal upgrade may need these helpers updated.

import json
import time
from types import SimpleNamespace

from src import config, methods
from src.schema import WebPage, to_html

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


def method(name):
    return getattr(modal_server.Generator._get_user_cls(), name)._get_raw_f()


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
    state = method("warm_state")(obj)
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
    record = method("run_guidance")(obj, "a bakery")
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
    record = method("run_guidance")(generator(), "a bakery")
    assert (record["bucket"], record["html"], record["hit_cap"]) == (
        "truncated",
        None,
        True,
    )


def test_method_b_sends_the_schema_prompt_and_assembles_html_from_valid_json():
    obj, prompts = generator(), []
    obj.complete = lambda prompt: (prompts.append(prompt), GOOD_JSON, 30, False)[1:]
    record = method("run_coercion")(obj, "a bakery")
    assert prompts == [methods.build_prompt("a bakery")]
    assert (record["method"], record["bucket"]) == ("coercion", "valid")
    assert record["html"] == to_html(WebPage.model_validate_json(GOOD_JSON))
    assert record["text"] == GOOD_JSON


def test_method_b_markdown_fence_is_malformed_and_has_no_html():
    obj = generator()
    obj.complete = lambda prompt: (f"```json\n{GOOD_JSON}\n```", 30, False)
    record = method("run_coercion")(obj, "a bakery")
    assert (record["bucket"], record["html"]) == ("malformed", None)


class FakeModal:
    """Stands in for the Modal class in the entrypoints: every remote call returns a full record."""

    def __init__(self):
        full = lambda m: lambda request: {
            "method": m, "bucket": "valid", "reason": "", "html": "<html></html>", "text": "x",
            "new_tokens": 1, "hit_cap": False, "seconds": 1.0,
        }  # fmt: skip
        self.probe = SimpleNamespace(remote=lambda: {"gpu": "Fake GPU"})
        self.warm_state = SimpleNamespace(remote=lambda: {"guidance": 1, "coercion": 1})
        self.run_guidance = SimpleNamespace(remote=full("guidance"))
        self.run_coercion = SimpleNamespace(remote=full("coercion"))


def test_run_writes_one_record_per_prompt_per_method_and_never_overwrites(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(modal_server, "Generator", FakeModal)
    run = modal_server.run.info.raw_f
    run("t1")
    data = json.loads((tmp_path / "results/benchmark_t1.json").read_text())
    assert len(data["records"]) == 2 * len(config.PROMPTS)
    assert data["probe"]["gpu"] == "Fake GPU" and data["warmups"] == []
    with pytest.raises(SystemExit):
        run("t1")


def test_smoke_runs_two_prompts_through_both_methods(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(modal_server, "Generator", FakeModal)
    modal_server.smoke.info.raw_f()
    data = json.loads((tmp_path / modal_server.PROBE_PATH).read_text())
    assert len(data["records"]) == 4 and data["probe"]["gpu"] == "Fake GPU"
