"""Frontend/inference tests. Ollama is always mocked: no server, GPU, or models needed."""

import subprocess
import sys
from types import SimpleNamespace
from typing import ClassVar

import gradio as gr
import ollama
import pytest

from src import frontend

QWEN25 = "qwen2.5:3b-instruct-q4_K_M"
QWEN35 = "qwen3.5:4b-q4_K_M"
HOST = "http://h:1"


def chunk(content, thinking=None, **stats):
    """Streamed chunk; ``stats`` are the timing fields only present on the final chunk."""
    fields = {
        "done": False,
        "eval_count": None,
        "eval_duration": None,
        "total_duration": None,
    }
    return SimpleNamespace(
        message=SimpleNamespace(content=content, thinking=thinking),
        **{**fields, **stats},
    )


def final_chunk(content="!"):
    """Last chunk: 120 tokens generated in 2 s (60 tokens/s), 2.5 s in total."""
    return chunk(
        content,
        done=True,
        eval_count=120,
        eval_duration=2_000_000_000,
        total_duration=2_500_000_000,
    )


class FakeClient:
    """Stand-in for ollama.Client that records calls and replays scripted chunks."""

    instances: ClassVar[list] = []
    chunks: ClassVar[object] = [chunk("Hel"), chunk("lo"), final_chunk()]
    chat_error: ClassVar[Exception | None] = None
    running: ClassVar[list] = []

    def __init__(self, host=None):
        self.host = host
        self.chat_calls = []
        self.generate_calls = []
        FakeClient.instances.append(self)

    def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        if FakeClient.chat_error:
            raise FakeClient.chat_error
        return iter(FakeClient.chunks)

    def ps(self):
        return SimpleNamespace(
            models=[SimpleNamespace(model=name) for name in FakeClient.running]
        )

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)


def no_nvidia_smi(*args, **kwargs):
    raise FileNotFoundError("nvidia-smi")


@pytest.fixture(autouse=True)
def fake_ollama(monkeypatch):
    FakeClient.instances = []
    FakeClient.chunks = [chunk("Hel"), chunk("lo"), final_chunk()]
    FakeClient.chat_error = None
    FakeClient.running = []
    monkeypatch.setattr(frontend.ollama, "Client", FakeClient)
    monkeypatch.setattr(
        frontend.subprocess, "run", no_nvidia_smi
    )  # same on every machine
    return FakeClient


def run_chat(
    message, history, model=QWEN25, think=False, host=frontend.DEFAULT_OLLAMA_HOST
):
    return list(frontend.chat(message, history, model, think, host))


def last_chat_call():
    return FakeClient.instances[-1].chat_calls[-1]


# --- configuration -----------------------------------------------------------------------


def test_default_model_is_qwen25_and_both_models_offered():
    values = [value for _, value in frontend.MODEL_CHOICES]
    assert frontend.DEFAULT_MODEL == QWEN25
    assert values == [QWEN25, QWEN35]


def test_default_host_and_context_size():
    assert frontend.DEFAULT_OLLAMA_HOST == "http://127.0.0.1:11434"
    assert frontend.NUM_CTX == 4096


def test_thinking_support_only_for_qwen35():
    assert frontend.supports_thinking(QWEN35)
    assert not frontend.supports_thinking(QWEN25)
    assert not frontend.supports_thinking("unknown")


# --- history formatting ------------------------------------------------------------------


def test_clean_history_drops_metadata_and_malformed_entries():
    history = [
        {"role": "user", "content": "hi", "metadata": {"title": "x"}, "options": []},
        {"role": "assistant", "content": "hello"},
        {"role": "system", "content": "ignored"},
        {"role": "user", "content": "   "},
        {"role": "user", "content": ["not", "text"]},
        {"role": "assistant"},
        ("user", "tuple"),
        None,
    ]
    assert frontend.clean_history(history) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


@pytest.mark.parametrize("bad", [None, "text", {"role": "user"}, 5])
def test_clean_history_rejects_non_lists(bad):
    assert frontend.clean_history(bad) == []


# --- dispatch, streaming, history --------------------------------------------------------


def test_stream_assembles_reply_and_commits_exchange():
    updates = run_chat("Hi", [])
    assert [u[0][-1]["content"] for u in updates][:3] == ["Hi", "Hel", "Hello"]
    assert updates[0][0] == [{"role": "user", "content": "Hi"}]
    chatbot, context, textbox = updates[-1]
    # the committed history keeps the stats; the chat view shows them underneath the reply
    assert context == [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "content": "Hello!",
            "stats": "60.0 tokens/s (120 tokens) · 2.5 s total",
        },
    ]
    assert chatbot[-1]["content"] == (
        'Hello!\n\n<small class="reply-stats">60.0 tokens/s (120 tokens) · 2.5 s total</small>'
    )
    assert textbox == gr.skip()  # leave what the user is typing alone


def test_textbox_cleared_only_on_first_update():
    updates = run_chat("Hi", [])
    assert updates[0][2] == ""
    assert all(textbox == gr.skip() for _, _, textbox in updates[1:])


def test_context_is_not_committed_until_reply_finishes():
    updates = run_chat("Hi", [])
    assert all(context == [] for _, context, _ in updates[:-1])


@pytest.mark.parametrize("model", [QWEN25, QWEN35])
def test_request_goes_to_selected_model(model):
    run_chat("Hi", [], model=model)
    call = last_chat_call()
    assert call["model"] == model
    assert call["stream"] is True
    assert call["options"] == {"num_ctx": 4096}


def test_client_uses_configured_host():
    run_chat("Hi", [], host="http://example:1234")
    assert FakeClient.instances[-1].host == "http://example:1234"


def test_history_is_prepended_to_latest_user_message():
    history = [
        {"role": "user", "content": "My name is Ada.", "metadata": {"x": 1}},
        {"role": "assistant", "content": "Nice to meet you."},
    ]
    run_chat("  What is my name?  ", history)
    assert last_chat_call()["messages"] == [
        {"role": "user", "content": "My name is Ada."},
        {"role": "assistant", "content": "Nice to meet you."},
        {"role": "user", "content": "What is my name?"},
    ]


def test_multi_turn_conversation_accumulates():
    FakeClient.chunks = [chunk("one")]
    history = run_chat("first", [])[-1][1]
    FakeClient.chunks = [chunk("two")]
    history = run_chat("second", history)[-1][1]
    assert [m["content"] for m in history] == ["first", "one", "second", "two"]
    assert [m["role"] for m in last_chat_call()["messages"]] == [
        "user",
        "assistant",
        "user",
    ]


def test_none_and_empty_chunks_are_ignored():
    FakeClient.chunks = [chunk(None), chunk("A"), chunk(""), chunk("B")]
    assert run_chat("Hi", [])[-1][0][-1]["content"] == "AB"


# --- speed readout and per-reply stats ---------------------------------------------------


@pytest.mark.parametrize(
    ("count", "eval_ns", "total_ns", "expected"),
    [
        (120, 2_000_000_000, 2_500_000_000, "60.0 tokens/s (120 tokens) · 2.5 s total"),
        (67, 1_000_000_000, None, "67.0 tokens/s (67 tokens)"),
        (None, None, 1_234_000_000, "1.2 s total"),
        (None, 1_000_000_000, 3_000_000_000, "3.0 s total"),
        (0, 1_000_000_000, 0, ""),
        (10, 0, None, ""),
        (None, None, None, ""),
    ],
)
def test_format_stats(count, eval_ns, total_ns, expected):
    assert frontend.format_stats(count, eval_ns, total_ns) == expected


def test_stats_are_attached_to_their_own_reply():
    history = run_chat("first", [])[-1][1]
    FakeClient.chunks = [chunk("two"), final_chunk("!")]  # same 60 tokens/s
    history = run_chat("second", history)[-1][1]
    first_stats = history[1]["stats"]
    assert (
        history[3]["stats"] == first_stats == "60.0 tokens/s (120 tokens) · 2.5 s total"
    )
    FakeClient.chunks = [
        chunk("three"),
        chunk(
            "!",
            done=True,
            eval_count=5,
            eval_duration=1_000_000_000,
            total_duration=1_000_000_000,
        ),
    ]
    history = run_chat("third", history)[-1][1]
    assert history[1]["stats"] == first_stats  # earlier replies keep their own stats
    assert history[5]["stats"] == "5.0 tokens/s (5 tokens) · 1.0 s total"


def test_stats_are_never_sent_back_to_ollama():
    history = run_chat("first", [])[-1][1]
    run_chat("second", history)
    assert all(set(m) == {"role", "content"} for m in last_chat_call()["messages"])


def test_stats_ignored_until_final_chunk():
    FakeClient.chunks = [chunk("Hi", eval_count=5, eval_duration=1, total_duration=1)]
    assert "stats" not in run_chat("Hi", [])[-1][1][-1]
    assert "reply-stats" not in run_chat("Hi", [])[-1][0][-1]["content"]


def test_no_stats_when_ollama_reports_none():
    FakeClient.chunks = [chunk("Hi there")]
    history = run_chat("Hi", [])[-1][1]
    assert "stats" not in history[-1]
    assert run_chat("Hi", [])[-1][0][-1] == {"role": "assistant", "content": "Hi there"}


def test_failed_turn_keeps_previous_replies_stats():
    history = run_chat("first", [])[-1][1]
    FakeClient.chat_error = ConnectionError("x")
    context = run_chat("second", history)[-1][1]
    assert context == history
    assert "stats" in context[-1]


def test_unknown_model_keeps_context_unchanged():
    history = run_chat("first", [])[-1][1]
    assert run_chat("Hi", history, model="gpt-9")[-1][1] == history


def test_blank_message_keeps_context_unchanged():
    history = run_chat("first", [])[-1][1]
    assert run_chat("  ", history)[-1][1] == history


def test_display_messages_adds_small_stats_only_to_replies_with_stats():
    messages = [
        {"role": "user", "content": "Hi", "stats": "ignored"},
        {"role": "assistant", "content": "Hello", "stats": "1.0 s total"},
        {"role": "assistant", "content": "No stats"},
    ]
    assert frontend.display_messages(messages) == [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "content": 'Hello\n\n<small class="reply-stats">1.0 s total</small>',
        },
        {"role": "assistant", "content": "No stats"},
    ]


def test_clean_history_keeps_stats_only_on_replies():
    history = [
        {"role": "user", "content": "Hi", "stats": "x"},
        {"role": "assistant", "content": "Hello", "stats": "1.0 s total"},
        {"role": "assistant", "content": "Empty stats", "stats": ""},
    ]
    assert frontend.clean_history(history) == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello", "stats": "1.0 s total"},
        {"role": "assistant", "content": "Empty stats"},
    ]


# --- thinking mode -----------------------------------------------------------------------


def test_thinking_enabled_for_qwen35():
    run_chat("Hi", [], model=QWEN35, think=True)
    assert last_chat_call()["think"] is True


def test_thinking_explicitly_disabled_by_default_for_qwen35():
    run_chat("Hi", [], model=QWEN35, think=False)
    assert last_chat_call()["think"] is False


@pytest.mark.parametrize("think", [True, False])
def test_thinking_never_sent_for_qwen25(think):
    run_chat("Hi", [], model=QWEN25, think=think)
    assert "think" not in last_chat_call()


def test_reasoning_text_is_not_displayed():
    FakeClient.chunks = [
        chunk("", thinking="secret plan"),
        chunk("Answer", thinking=None),
    ]
    chatbot, context, _ = run_chat("Hi", [], model=QWEN35, think=True)[-1]
    assert chatbot[-1]["content"] == "Answer"
    assert "secret" not in str(context)


# --- model switching and new chats -----------------------------------------------------


def switch(model, history=None, previous=QWEN25, archive=None, host=HOST):
    return frontend.switch_model(model, history or [], previous, archive or [], host)


def test_switch_model_resets_chat_and_thinking():
    chatbot, context, checkbox, archive, model = switch(QWEN35)
    assert (chatbot, context, archive, model) == ([], [], [], QWEN35)
    assert checkbox["value"] is False
    assert checkbox["interactive"] is True


def test_switch_to_qwen25_disables_thinking_control():
    checkbox = switch(QWEN25, previous=QWEN35)[2]
    assert checkbox["value"] is False
    assert checkbox["interactive"] is False


def test_switch_model_moves_current_chat_to_earlier_chats():
    old = {"title": "old", "model": QWEN25, "messages": []}
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello", "stats": "1.0 s total"},
    ]
    chatbot, context, _, archive, model = switch(
        QWEN35, history, previous=QWEN25, archive=[old]
    )
    assert (chatbot, context, model) == ([], [], QWEN35)
    assert archive == [old, {"title": "Hi", "model": QWEN25, "messages": history}]


def test_switch_model_with_empty_chat_archives_nothing():
    assert switch(QWEN35)[3] == []


def test_switch_model_unloads_other_running_models_only():
    FakeClient.running = [QWEN25, QWEN35]
    switch(QWEN35)
    client = FakeClient.instances[-1]
    assert client.host == HOST
    assert client.generate_calls == [{"model": QWEN25, "prompt": "", "keep_alive": 0}]


def test_switch_model_does_nothing_when_nothing_running():
    switch(QWEN35)
    assert FakeClient.instances[-1].generate_calls == []


def test_unload_failure_is_ignored(monkeypatch):
    def boom(self):
        raise ConnectionError("down")

    monkeypatch.setattr(FakeClient, "ps", boom)
    assert switch(QWEN25)[:2] == ([], [])


def test_start_new_chat_resets_view_and_model_context():
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    chatbot, context, box, archive = frontend.start_new_chat(history, QWEN25, [])
    assert (chatbot, context, box) == ([], [], "")
    assert archive == [{"title": "Hi", "model": QWEN25, "messages": history}]
    # The next turn sends none of the earlier messages to the model.
    run_chat("fresh start", context)
    assert last_chat_call()["messages"] == [{"role": "user", "content": "fresh start"}]


def test_start_new_chat_on_empty_chat_archives_nothing():
    assert frontend.start_new_chat([], QWEN25, [])[3] == []


# --- invalid input -----------------------------------------------------------------------


@pytest.mark.parametrize("message", ["", "   \n\t", None, 42])
def test_blank_or_invalid_message_does_not_call_ollama(message):
    history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    updates = run_chat(message, history)
    assert updates == [(history, history, "")]
    assert FakeClient.instances == []


def test_unknown_model_is_reported_without_calling_ollama():
    chatbot, context, _ = run_chat("Hi", [], model="gpt-9")[-1]
    assert "Unknown model" in chatbot[-1]["content"]
    assert context == []
    assert FakeClient.instances == []


# --- failures ----------------------------------------------------------------------------


def failed_turn(error, model=QWEN25, history=None):
    FakeClient.chat_error = error
    history = history or []
    chatbot, context, _ = run_chat("Hi", history, model=model)[-1]
    return chatbot, context


def test_connection_failure_is_shown_and_not_committed():
    chatbot, context = failed_turn(ConnectionError("refused"))
    assert "Cannot reach Ollama" in chatbot[-1]["content"]
    assert "http://127.0.0.1:11434" in chatbot[-1]["content"]
    assert context == []


def test_missing_model_message_suggests_pull():
    chatbot, _ = failed_turn(ollama.ResponseError("not found", 404), model=QWEN35)
    assert f"ollama pull {QWEN35}" in chatbot[-1]["content"]


def test_other_ollama_error_is_reported():
    chatbot, _ = failed_turn(ollama.ResponseError("out of memory", 500))
    assert "out of memory" in chatbot[-1]["content"]


def test_unexpected_error_is_reported():
    chatbot, _ = failed_turn(RuntimeError("boom"))
    assert "Inference failed: boom" in chatbot[-1]["content"]


def test_failure_keeps_previous_context():
    history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    chatbot, context = failed_turn(ConnectionError("x"), history=history)
    assert context == history
    assert chatbot[:2] == history


def test_stream_interrupted_midway_is_not_committed():
    class Broken:
        def __iter__(self):
            yield chunk("partial")
            raise ConnectionError("dropped")

    FakeClient.chunks = Broken()
    updates = run_chat("Hi", [])
    assert updates[-2][0][-1]["content"] == "partial"
    assert "Cannot reach Ollama" in updates[-1][0][-1]["content"]
    assert updates[-1][1] == []


@pytest.mark.parametrize("chunks", [[], [chunk("")], [chunk("  \n")]])
def test_empty_reply_is_an_error(chunks):
    FakeClient.chunks = chunks
    chatbot, context, _ = run_chat("Hi", [])[-1]
    assert "empty response" in chatbot[-1]["content"]
    assert context == []


def test_thinking_shows_placeholder_until_answer_starts():
    FakeClient.chunks = [chunk("", thinking="hmm"), chunk("Hi"), final_chunk("!")]
    updates = run_chat("Hi", [], model=QWEN35, think=True)
    shown = [
        u[0][-1]["content"] for u in updates[1:-1]
    ]  # skip the user-only and final updates
    assert shown == ["*Thinking…*", "Hi", "Hi!"]  # placeholder until the answer starts
    assert updates[-1][1][-1] == {
        "role": "assistant",
        "content": "Hi!",
        "stats": "60.0 tokens/s (120 tokens) · 2.5 s total",
    }
    assert "Thinking" not in str(updates[-1][1])  # placeholder never saved to history


# --- Gradio wiring -----------------------------------------------------------------------


def components(demo, kind):
    return [b for b in demo.blocks.values() if isinstance(b, kind)]


def handler(demo, name):
    return next(
        f.fn for f in demo.fns.values() if f.fn is not None and f.fn.__name__ == name
    )


def test_demo_has_required_components():
    demo = frontend.build_demo()
    (dropdown,) = components(demo, gr.Dropdown)
    (checkbox,) = components(demo, gr.Checkbox)
    (textbox,) = components(demo, gr.Textbox)
    assert components(demo, gr.Chatbot)
    assert {m.elem_id for m in components(demo, gr.Markdown) if m.elem_id} == {
        "app-header"
    }
    assert [value for _, value in dropdown.choices] == [QWEN25, QWEN35]
    assert dropdown.value == QWEN25
    assert checkbox.value is False
    assert checkbox.interactive is False
    assert textbox.submit_btn
    # "Clear conversation" became "Start new chat"
    assert [b.value for b in components(demo, gr.Button)] == ["＋  Start new chat"]
    assert len(components(demo, gr.Sidebar)) == 1


def test_demo_respond_uses_configured_host():
    demo = frontend.build_demo("http://remote:9")
    updates = list(handler(demo, "respond")("Hi", [], QWEN25, False))
    assert updates[-1][1][-1]["content"] == "Hello!"
    assert FakeClient.instances[-1].host == "http://remote:9"


def test_demo_model_change_handler_archives_and_resets_state():
    demo = frontend.build_demo("http://remote:9")
    history = [{"role": "user", "content": "Hi"}]
    chatbot, context, checkbox, archive, model = handler(demo, "on_model_change")(
        QWEN35, history, QWEN25, []
    )
    assert (chatbot, context, checkbox["interactive"]) == ([], [], True)
    assert archive == [{"title": "Hi", "model": QWEN25, "messages": history}]
    assert model == QWEN35
    assert FakeClient.instances[-1].host == "http://remote:9"


# --- launcher ----------------------------------------------------------------------------


class FakeDemo:
    def __init__(self):
        self.launch_kwargs = None

    def launch(self, **kwargs):
        self.launch_kwargs = kwargs


@pytest.fixture
def launched(monkeypatch):
    demo = FakeDemo()
    hosts = []
    monkeypatch.setattr(frontend, "build_demo", lambda host: hosts.append(host) or demo)
    return demo, hosts


def test_main_defaults_to_localhost_7860(monkeypatch, launched):
    for name in ("OLLAMA_HOST", "GRADIO_SERVER_NAME", "GRADIO_SERVER_PORT"):
        monkeypatch.delenv(name, raising=False)
    frontend.main()
    demo, hosts = launched
    assert hosts == ["http://127.0.0.1:11434"]
    assert demo.launch_kwargs["server_name"] == "127.0.0.1"
    assert demo.launch_kwargs["server_port"] == 7860


def test_main_reads_environment(monkeypatch, launched):
    monkeypatch.setenv("OLLAMA_HOST", "http://other:11434")
    monkeypatch.setenv("GRADIO_SERVER_NAME", "127.0.0.2")
    monkeypatch.setenv("GRADIO_SERVER_PORT", "9000")
    frontend.main()
    demo, hosts = launched
    assert hosts == ["http://other:11434"]
    assert demo.launch_kwargs["server_name"] == "127.0.0.2"
    assert demo.launch_kwargs["server_port"] == 9000


# --- earlier chats sidebar ---------------------------------------------------------------


def entry(title, model, text):
    """An archived chat with one exchange whose first user message is ``text``."""
    return {
        "title": title,
        "model": model,
        "messages": [
            {"role": "user", "content": text},
            {"role": "assistant", "content": f"re: {text}"},
        ],
    }


def test_chat_title_is_short_single_line_and_truncated():
    messages = [
        {
            "role": "user",
            "content": "  Explain   quantum\nentanglement simply please, thanks a lot  ",
        }
    ]
    assert frontend.chat_title(messages) == "Explain quantum entanglement simply plea…"
    assert frontend.chat_title([{"role": "user", "content": "Hi"}]) == "Hi"


def test_chat_label_shows_title_and_short_model_name():
    assert frontend.chat_label(entry("Hi", QWEN25, "Hi")) == "Hi\nqwen2.5"
    assert frontend.chat_label(entry("Yo", QWEN35, "Yo")) == "Yo\nqwen3.5"


def test_archive_chat_keeps_only_finished_non_empty_conversations():
    assert frontend.archive_chat([], [], QWEN25) == []
    messages = [
        {"role": "user", "content": "Hi", "metadata": {"x": 1}},
        {"role": "assistant", "content": "Hello"},
    ]
    assert frontend.archive_chat([], messages, QWEN35) == [
        {
            "title": "Hi",
            "model": QWEN35,
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
        }
    ]


def test_archive_chat_does_not_mutate_its_input():
    first = [entry("old", QWEN25, "old")]
    result = frontend.archive_chat(first, [{"role": "user", "content": "new"}], QWEN25)
    assert len(first) == 1
    assert [item["title"] for item in result] == ["old", "new"]


def test_demo_start_new_chat_handler_archives_current_chat():
    demo = frontend.build_demo()
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    result = handler(demo, "start_new_chat")(history, QWEN25, [])
    assert result[:3] == ([], [], "")
    assert result[3] == [{"title": "Hi", "model": QWEN25, "messages": history}]


def test_open_archived_chat_restores_it_and_archives_the_current_one():
    FakeClient.running = [QWEN25]
    archive = [entry("Old", QWEN35, "old question")]
    current = [
        {"role": "user", "content": "now"},
        {"role": "assistant", "content": "ok", "stats": "1.0 s total"},
    ]
    chatbot, history, box, new_archive, model, think, chat_model = (
        frontend.open_archived_chat(0, current, QWEN25, archive, HOST)
    )
    assert chatbot == frontend.display_messages(archive[0]["messages"])
    assert history == archive[0]["messages"]
    assert box == gr.skip()
    assert model == {"__type__": "update", "value": QWEN35}
    assert chat_model == QWEN35
    assert think["value"] is False and think["interactive"] is True
    assert new_archive == [{"title": "now", "model": QWEN25, "messages": current}]
    # the model used by the current chat is unloaded; the opened chat's model is kept
    assert FakeClient.instances[-1].generate_calls == [
        {"model": QWEN25, "prompt": "", "keep_alive": 0}
    ]
    assert FakeClient.instances[-1].host == HOST


def test_open_archived_chat_with_empty_current_does_not_archive_it():
    archive = [entry("Old", QWEN25, "old question")]
    result = frontend.open_archived_chat(0, [], QWEN25, archive, HOST)
    assert result[3] == []


@pytest.mark.parametrize("index", [None, -1, 1, "0", True])
def test_open_archived_chat_ignores_invalid_index(index):
    archive = [entry("Old", QWEN25, "q")]
    history = [{"role": "user", "content": "x"}]
    result = frontend.open_archived_chat(index, history, QWEN35, archive, HOST)
    assert result[0] == result[2] == gr.skip()
    assert result[1] is history and result[3] is archive and result[6] == QWEN35
    assert FakeClient.instances == []


def test_delete_archived_chat_removes_only_that_chat():
    archive = [
        entry("a", QWEN25, "a"),
        entry("b", QWEN35, "b"),
        entry("c", QWEN25, "c"),
    ]
    assert frontend.delete_archived_chat(1, archive) == [archive[0], archive[2]]
    assert len(archive) == 3  # input untouched


@pytest.mark.parametrize("index", [None, -1, 3, "0", True])
def test_delete_archived_chat_ignores_invalid_index(index):
    archive = [entry("a", QWEN25, "a")]
    assert frontend.delete_archived_chat(index, archive) is archive


def render_rows(archive, host=HOST):
    """Run the gr.render callback like Gradio does; return the demo and the new handlers."""
    demo = frontend.build_demo(host)
    renderable = next(
        f.renderable for f in demo.fns.values() if getattr(f, "renderable", None)
    )
    before = set(demo.fns)
    with demo:
        renderable.fn(archive)
    handlers = [f.fn for k, f in demo.fns.items() if k not in before and f.fn]
    return demo, handlers


def test_empty_earlier_chats_list_shows_placeholder():
    demo, handlers = render_rows([])
    texts = [m.value for m in components(demo, gr.Markdown)]
    assert "No earlier chats yet." in texts
    assert handlers == []


def test_earlier_chats_list_has_open_and_delete_button_per_chat_newest_first():
    archive = [entry("Hi", QWEN25, "Hi"), entry("Yo", QWEN35, "Yo")]
    demo, handlers = render_rows(archive)
    labels = [b.value for b in components(demo, gr.Button)]
    assert labels == ["＋  Start new chat", "Yo\nqwen3.5", "🗑️", "Hi\nqwen2.5", "🗑️"]
    assert [h.__name__ for h in handlers] == ["open_this", "delete_this"] * 2


def test_earlier_chat_buttons_act_on_their_own_chat():
    archive = [entry("Hi", QWEN25, "Hi"), entry("Yo", QWEN35, "Yo")]
    _, (open_newest, delete_newest, open_oldest, delete_oldest) = render_rows(archive)
    current = [{"role": "user", "content": "now"}]

    opened = open_newest(current, QWEN25, archive)
    assert opened[0] == frontend.display_messages(archive[1]["messages"])
    assert opened[6] == QWEN35
    assert [e["title"] for e in opened[3]] == ["Hi", "now"]
    assert FakeClient.instances[-1].host == HOST

    assert open_oldest([], QWEN35, archive)[0] == frontend.display_messages(
        archive[0]["messages"]
    )
    assert delete_newest(archive) == [archive[0]]
    assert delete_oldest(archive) == [archive[1]]


# --- thinking hint, header and environment -----------------------------------------------


def test_thinking_update_hint_depends_on_model():
    qwen25 = frontend.thinking_update(QWEN25)
    assert (qwen25["value"], qwen25["interactive"], qwen25["info"]) == (
        False,
        False,
        "Not supported",
    )
    qwen35 = frontend.thinking_update(QWEN35)
    assert (qwen35["value"], qwen35["interactive"], qwen35["info"]) == (
        False,
        True,
        "Slower",
    )


def test_switching_model_updates_thinking_hint():
    assert switch(QWEN35)[2]["info"] == "Slower"
    assert switch(QWEN25, previous=QWEN35)[2]["info"] == "Not supported"


def test_opening_archived_chat_updates_thinking_hint():
    archive = [entry("Old", QWEN35, "q")]
    assert (
        frontend.open_archived_chat(0, [], QWEN25, archive, HOST)[5]["info"] == "Slower"
    )


def test_demo_thinking_checkbox_starts_with_not_available_hint():
    (checkbox,) = components(frontend.build_demo(), gr.Checkbox)
    assert checkbox.info == "Not supported"


def fake_nvidia_smi(stdout):
    def run(command, **kwargs):
        assert command[0] == "nvidia-smi"
        return SimpleNamespace(stdout=stdout)

    return run


def test_detect_gpus_formats_name_and_memory(monkeypatch):
    monkeypatch.setattr(
        frontend.subprocess, "run", fake_nvidia_smi("Quadro T2000, 4096\n")
    )
    assert frontend.detect_gpus() == "Quadro T2000 (4 GB)"


def test_detect_gpus_lists_every_gpu(monkeypatch):
    output = "Tesla T4, 15360\nTesla T4, 15360\n"
    monkeypatch.setattr(frontend.subprocess, "run", fake_nvidia_smi(output))
    assert frontend.detect_gpus() == "Tesla T4 (15 GB), Tesla T4 (15 GB)"


def test_detect_gpus_tolerates_unparsable_memory(monkeypatch):
    monkeypatch.setattr(frontend.subprocess, "run", fake_nvidia_smi("Odd GPU, [N/A]"))
    assert frontend.detect_gpus() == "Odd GPU"


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError("nvidia-smi"), subprocess.CalledProcessError(9, "nvidia-smi")]
    + [subprocess.TimeoutExpired("nvidia-smi", 5)],
)
def test_detect_gpus_without_nvidia_smi(monkeypatch, error):
    def boom(*args, **kwargs):
        raise error

    monkeypatch.setattr(frontend.subprocess, "run", boom)
    assert frontend.detect_gpus() == "No GPU detected"


def test_detect_gpus_with_empty_output(monkeypatch):
    monkeypatch.setattr(frontend.subprocess, "run", fake_nvidia_smi(""))
    assert frontend.detect_gpus() == "No GPU detected"


def test_is_colab_detects_env_var_and_module(monkeypatch):
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    assert not frontend.is_colab()
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-colab_20260101")
    assert frontend.is_colab()
    monkeypatch.delenv("COLAB_RELEASE_TAG")
    monkeypatch.setitem(sys.modules, "google.colab", object())
    assert frontend.is_colab()


def test_banner_for_local_machine_shows_gpu_only(monkeypatch):
    monkeypatch.setattr(frontend, "is_colab", lambda: False)
    monkeypatch.setattr(frontend, "detect_gpus", lambda: "Quadro T2000 (4 GB)")
    assert frontend.environment_banner() == (
        "Assignment 11 - Running Locally",
        "Quadro T2000 (4 GB)",
    )


def test_banner_for_colab_shows_colab_and_gpu(monkeypatch):
    monkeypatch.setattr(frontend, "is_colab", lambda: True)
    monkeypatch.setattr(frontend, "detect_gpus", lambda: "Tesla T4 (15 GB)")
    assert frontend.environment_banner() == (
        "Assignment 11 - Running on Colab",
        "Tesla T4 (15 GB)",
    )


def test_demo_header_and_tab_title_come_from_the_banner(monkeypatch):
    monkeypatch.setattr(frontend, "environment_banner", lambda: ("Title X", "Detail Y"))
    demo = frontend.build_demo()
    header = next(m for m in components(demo, gr.Markdown) if m.elem_id == "app-header")
    assert header.value == "# Title X\nDetail Y"
    assert demo.title == "Title X"
