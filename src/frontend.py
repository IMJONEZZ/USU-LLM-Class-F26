"""Gradio chat frontend backed by a local Ollama server."""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from collections.abc import Iterator
from typing import Any

import gradio as gr
import ollama

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_SERVER_NAME = "127.0.0.1"
DEFAULT_SERVER_PORT = 7860
NUM_CTX = 4096

DEFAULT_MODEL = "qwen2.5:3b-instruct-q4_K_M"
THINKING_MODEL = "qwen3.5:4b-q4_K_M"
MODEL_CHOICES = [
    ("Qwen2.5-3B-Instruct (fast, default)", DEFAULT_MODEL),
    ("Qwen3.5-4B (optional thinking)", THINKING_MODEL),
]
ASSIGNMENT_NAME = "Assignment 11"
THINKING_INFO = {True: "Slower", False: "Not supported"}
KNOWN_MODELS = frozenset(value for _, value in MODEL_CHOICES)
THINKING_MODELS = frozenset({THINKING_MODEL})

Message = dict[str, str]


def supports_thinking(model: str) -> bool:
    """Return whether Ollama's ``think`` option may be used with ``model``."""
    return model in THINKING_MODELS


def thinking_update(model: str) -> Any:
    """Checkbox update: off, enabled only for thinking models, with a matching hint underneath."""
    available = supports_thinking(model)
    return gr.update(value=False, interactive=available, info=THINKING_INFO[available])


def is_colab() -> bool:
    """Whether the app is running inside Google Colab."""
    return "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules


def detect_gpus() -> str:
    """Name and memory of the NVIDIA GPU(s) reported by ``nvidia-smi``."""
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except OSError, subprocess.SubprocessError:
        return "No GPU detected"
    gpus = []
    for line in output.strip().splitlines():
        name, _, memory_mib = line.partition(",")
        try:
            gpus.append(f"{name.strip()} ({round(int(memory_mib) / 1024)} GB)")
        except ValueError:
            gpus.append(name.strip())
    return ", ".join(gpus) or "No GPU detected"


def environment_banner() -> tuple[str, str]:
    """Header ``(title, detail)``: where the app runs (Colab or locally) and its GPU."""
    gpus = detect_gpus()
    if is_colab():
        return f"{ASSIGNMENT_NAME} - Running on Colab", gpus
    return f"{ASSIGNMENT_NAME} - Running Locally", gpus


def clean_history(history: Any) -> list[dict]:
    """Keep only well-formed plain-text user/assistant turns (plus each reply's stats)."""
    if not isinstance(history, list):
        return []
    cleaned = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role, content = item.get("role"), item.get("content")
        if (
            role in ("user", "assistant")
            and isinstance(content, str)
            and content.strip()
        ):
            entry = {"role": role, "content": content}
            stats = item.get("stats")
            if role == "assistant" and isinstance(stats, str) and stats:
                entry["stats"] = stats
            cleaned.append(entry)
    return cleaned


def to_ollama(messages: list[dict]) -> list[Message]:
    """Messages as Ollama expects them: role and content only."""
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def display_messages(messages: list[dict]) -> list[Message]:
    """Chatbot view: each reply's stats are shown in small text underneath that reply."""
    shown = []
    for m in messages:
        content = m["content"]
        if m["role"] == "assistant" and m.get("stats"):
            content += f'\n\n<small class="reply-stats">{m["stats"]}</small>'
        shown.append({"role": m["role"], "content": content})
    return shown


def unload_other_models(client: ollama.Client, keep: str) -> None:
    """Best-effort unload of every running Ollama model except ``keep`` (frees VRAM)."""
    with contextlib.suppress(Exception):  # housekeeping must never break the UI
        for running in client.ps().models:
            if running.model and running.model != keep:
                client.generate(model=running.model, prompt="", keep_alive=0)


def format_stats(
    token_count: int | None, eval_ns: int | None, total_ns: int | None
) -> str:
    """Format Ollama's stats, e.g. ``66.9 tokens/s (120 tokens) · 1.8 s total``.

    Either part is omitted when Ollama did not report it. The total time is Ollama's
    ``total_duration`` (it includes loading the model if it was not already in memory).
    """
    parts = []
    if token_count and eval_ns:
        parts.append(
            f"{token_count / eval_ns * 1e9:.1f} tokens/s ({token_count} tokens)"
        )
    if total_ns:
        parts.append(f"{total_ns / 1e9:.1f} s total")
    return " · ".join(parts)


def stream_reply(
    client: ollama.Client, model: str, messages: list[Message], think: bool
) -> Iterator[tuple[str, str]]:
    """Yield ``(reply so far, stats text)`` per chunk; stats are empty until the final chunk."""
    kwargs: dict[str, Any] = {"options": {"num_ctx": NUM_CTX}}
    if supports_thinking(model):
        kwargs["think"] = (
            think  # explicit False keeps Qwen3.5 from reasoning by default
        )
    reply = stats = ""
    for chunk in client.chat(model=model, messages=messages, stream=True, **kwargs):
        reply += (
            chunk.message.content or ""
        )  # reasoning (chunk.message.thinking) is not shown
        if chunk.done:  # only the final chunk carries timing stats
            stats = format_stats(
                chunk.eval_count, chunk.eval_duration, chunk.total_duration
            )
        yield reply, stats


class EmptyReplyError(Exception):
    """The model finished streaming without producing any text."""


def describe_error(error: Exception, model: str, host: str) -> str:
    """Turn an Ollama/transport failure into a short message for the chat window."""
    if isinstance(error, ConnectionError):
        return f"Cannot reach Ollama at {host}. Start it with `ollama serve`."
    if isinstance(error, ollama.ResponseError):
        if error.status_code == 404:
            return f"Model `{model}` is not available in Ollama. Run `ollama pull {model}`."
        return f"Ollama returned an error: {error.error}"
    if isinstance(error, EmptyReplyError):
        return f"Model `{model}` returned an empty response."
    return f"Inference failed: {error}"


def chat(
    message: str,
    history: list[dict],
    model: str,
    think: bool,
    host: str = DEFAULT_OLLAMA_HOST,
) -> Iterator[tuple[list[Message], list[dict], Any]]:
    """Stream one chat turn.

    Yields ``(chatbot, context, textbox)``. ``context`` is the clean history kept for the next
    turn, including each reply's stats. It only changes once the reply has completed, so failed or
    cancelled turns leave no trace. The textbox is cleared only on the first update, so text
    typed while a reply streams survives.
    """
    context = clean_history(history)
    if not isinstance(message, str) or not message.strip():
        yield display_messages(context), context, ""
        return
    user = {"role": "user", "content": message.strip()}
    if model not in KNOWN_MODELS:
        error = {"role": "assistant", "content": f"Unknown model `{model}`."}
        yield display_messages(context + [user, error]), context, ""
        return

    think = bool(think) and supports_thinking(model)
    yield display_messages(context + [user]), context, ""
    try:
        client = ollama.Client(host=host)
        reply = stats = ""
        # Until the first answer token arrives (Thinking mode reasons first), show a placeholder
        # instead of an empty bubble. It is only displayed, never saved to the history.
        placeholder = "*Thinking…*" if think else "…"
        for reply, stats in stream_reply(
            client, model, to_ollama(context + [user]), think
        ):
            partial = {"role": "assistant", "content": reply or placeholder}
            yield display_messages(context + [user, partial]), context, gr.skip()
        if not reply.strip():
            raise EmptyReplyError
    except Exception as error:  # noqa: BLE001 - shown to the user instead of crashing
        shown = {
            "role": "assistant",
            "content": f"⚠️ {describe_error(error, model, host)}",
        }
        yield display_messages(context + [user, shown]), context, gr.skip()
        return
    answer = {"role": "assistant", "content": reply}
    if stats:
        answer["stats"] = stats
    final = context + [user, answer]
    yield display_messages(final), final, gr.skip()


ARCHIVE_TITLE_LENGTH = 40


def chat_title(messages: list[Message]) -> str:
    """Short sidebar title taken from the first user message."""
    first = next((m["content"] for m in messages if m["role"] == "user"), "")
    text = " ".join(first.split())
    if len(text) > ARCHIVE_TITLE_LENGTH:
        return text[:ARCHIVE_TITLE_LENGTH] + "…"
    return text


def chat_label(item: dict) -> str:
    """Sidebar button text for an archived chat: its title, then the model it used."""
    return f"{item['title']}\n{item['model'].split(':')[0]}"


def archive_chat(archive: list[dict], messages: Any, model: str) -> list[dict]:
    """Return ``archive`` with a finished conversation appended; empty chats are not kept."""
    messages = clean_history(messages)
    if not messages:
        return list(archive)
    entry = {"title": chat_title(messages), "model": model, "messages": messages}
    return [*archive, entry]


def valid_index(index: Any, archive: list[dict]) -> bool:
    """Whether ``index`` points at an archived chat (bools and strings are rejected)."""
    return type(index) is int and 0 <= index < len(archive)


def switch_model(
    model: str,
    history: Any,
    previous_model: str,
    archive: list[dict],
    host: str = DEFAULT_OLLAMA_HOST,
) -> tuple[list, list, Any, list[dict], str]:
    """Move the current chat to Earlier chats, start fresh on ``model`` and free the old one.

    Thinking mode is reset to off and its checkbox enabled only for models that support it.
    """
    archive = archive_chat(archive, history, previous_model)
    unload_other_models(ollama.Client(host=host), keep=model)
    return [], [], thinking_update(model), archive, model


def start_new_chat(
    history: Any, model: str, archive: list[dict]
) -> tuple[list, list, str, list[dict]]:
    """Move the current chat to Earlier chats and start an empty one (fresh model context)."""
    return [], [], "", archive_chat(archive, history, model)


def delete_archived_chat(index: Any, archive: list[dict]) -> list[dict]:
    """Return ``archive`` without chat ``index``; an invalid index changes nothing."""
    if not valid_index(index, archive):
        return archive
    return [entry for i, entry in enumerate(archive) if i != index]


def open_archived_chat(
    index: Any, history: Any, model: str, archive: list[dict], host: str
) -> tuple:
    """Reopen archived chat ``index``; the chat that was open is archived in its place.

    Returns ``(chatbot, history, textbox, archive, model dropdown, thinking, model)``.
    """
    if not valid_index(index, archive):
        skip = gr.skip()
        return skip, history, skip, archive, skip, skip, model
    item = archive[index]
    remaining = [entry for i, entry in enumerate(archive) if i != index]
    remaining = archive_chat(remaining, history, model)
    unload_other_models(ollama.Client(host=host), keep=item["model"])
    messages = clean_history(item["messages"])
    return (
        display_messages(messages),
        messages,
        gr.skip(),
        remaining,
        gr.update(value=item["model"]),
        thinking_update(item["model"]),
        item["model"],
    )


def render_chat_list(
    archive: list[dict],
    *,
    open_chat: Any,
    open_inputs: list,
    open_outputs: list,
    archive_state: Any,
    cancels: list,
) -> None:
    """Draw the Earlier chats list (newest first): one open button and one delete button per chat.

    Runs inside ``gr.render``, which calls it again whenever the archive changes.
    """
    if not archive:
        gr.Markdown("No earlier chats yet.", elem_classes="empty-chats")
        return
    for index in reversed(range(len(archive))):
        with gr.Row(elem_classes="chat-row", equal_height=True):
            open_button = gr.Button(
                chat_label(archive[index]), elem_classes="chat-item", scale=8
            )
            delete_button = gr.Button(
                "🗑️", variant="stop", elem_classes="chat-delete", scale=1, min_width=44
            )

        def open_this(hist, model_name, archive_value, index=index):
            return open_chat(index, hist, model_name, archive_value)

        def delete_this(archive_value, index=index):
            return delete_archived_chat(index, archive_value)

        open_button.click(open_this, open_inputs, open_outputs, cancels=cancels)
        delete_button.click(delete_this, archive_state, archive_state)


THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="indigo",
    neutral_hue=gr.themes.colors.neutral,
    radius_size=gr.themes.sizes.radius_lg,
    font=[gr.themes.Font("system-ui"), "ui-sans-serif", "sans-serif"],
).set(
    body_background_fill="#f4f4f5",
    block_background_fill="#ffffff",
    block_border_color="#e4e4e7",
    # Dark mode is tuned for OLED screens: pure black surfaces and neutral grey accents
    # (no blue or purple tint). Light mode keeps the indigo accent.
    body_background_fill_dark="#000000",
    background_fill_primary_dark="#000000",
    background_fill_secondary_dark="#0b0b0b",
    block_background_fill_dark="#000000",
    block_border_color_dark="#262626",
    block_title_background_fill_dark="#1c1c1c",
    block_title_text_color_dark="#d4d4d4",
    border_color_primary_dark="#262626",
    border_color_accent_dark="#525252",
    color_accent_soft_dark="#1c1c1c",
    block_label_background_fill_dark="#1c1c1c",
    block_label_text_color_dark="#d4d4d4",
    input_background_fill_dark="#0b0b0b",
    input_border_color_dark="#262626",
    input_border_color_focus_dark="#737373",
    button_primary_background_fill_dark="#e5e5e5",
    button_primary_background_fill_hover_dark="#ffffff",
    button_primary_border_color_dark="#e5e5e5",
    button_primary_text_color_dark="#000000",
    button_secondary_background_fill_dark="#141414",
    button_secondary_background_fill_hover_dark="#262626",
    button_secondary_border_color_dark="#262626",
    button_secondary_text_color_dark="#e5e5e5",
    button_cancel_background_fill_dark="#141414",
    button_cancel_background_fill_hover_dark="#3f1212",
    button_cancel_border_color_dark="#262626",
    button_cancel_text_color_dark="#e5e5e5",
    checkbox_background_color_dark="#0b0b0b",
    checkbox_background_color_selected_dark="#e5e5e5",
    checkbox_border_color_dark="#525252",
    checkbox_border_color_selected_dark="#e5e5e5",
    checkbox_label_background_fill_dark="transparent",
    checkbox_label_background_fill_hover_dark="transparent",
    checkbox_label_background_fill_selected_dark="transparent",
    body_text_color_dark="#e5e5e5",
    body_text_color_subdued_dark="#a3a3a3",
)

CSS = """
#app { max-width: 880px; width: 100%; margin: 0 auto; }
#app-header { text-align: center; padding: 6px 0 2px; }
#app-header h1 { font-size: 1.7rem; font-weight: 700; margin: 0; }
#app-header p { margin: 4px 0 0; opacity: .65; }
#controls { align-items: flex-end; gap: 14px; }
#chat { border-radius: 16px; }
#chat .icon-button-wrapper:has(button[aria-label="Clear"]) { display: none; }
.reply-stats { display: block; margin-top: 6px; font-size: .75rem; opacity: .6; }
.sidebar-heading { opacity: .6; font-size: .75rem; text-transform: uppercase;
  letter-spacing: .06em; margin: 14px 2px 2px; }
.empty-chats { opacity: .55; font-size: .9rem; padding: 4px 2px; }
.chat-row { flex-wrap: nowrap !important; gap: 6px !important; align-items: center; }
.chat-item { justify-content: flex-start !important; text-align: left;
  font-weight: 400 !important; font-size: .85rem; line-height: 1.3;
  overflow: hidden; white-space: pre-line; max-height: 4.2em; }
.chat-item, .chat-delete { box-shadow: none !important; }
.chat-delete { min-width: 40px !important; padding: 0 !important; }
.chat-delete:hover { background: #fee2e2 !important; }
.dark * { scrollbar-color: #3a3a3a transparent; }
.dark .chat-delete { filter: grayscale(1); }
.dark .sidebar { background: #000 !important; border-right: 1px solid #1f1f1f; }
.message-buttons-left { display: flex !important; flex-direction: row !important;
  align-items: center; gap: 4px; width: auto !important; }
.message-buttons-left .icon-button-wrapper { width: auto !important; }
.message-buttons-left .reply-stats { display: inline-flex; align-items: center;
  margin: 0; font-size: .72rem; opacity: .6; white-space: nowrap; }
"""

# Gradio has no per-message footer, so the stats are moved (in the browser) from the bubble
# into the copy-button row beside it. Uses Gradio's chatbot class names; if this script does
# not run, the stats simply stay under the reply text.
STATS_HEAD = """
<script>
  (() => {
    const placeStats = () => {
      document.querySelectorAll("#chat .message-row.bot-row").forEach((row) => {
        const bar = row.nextElementSibling;
        const stats = row.querySelector(".reply-stats");
        if (!bar || !stats || !bar.classList.contains("message-buttons-left")) return;
        bar.querySelectorAll(".reply-stats").forEach((old) => old.remove());
        bar.appendChild(stats);
      });
    };
    new MutationObserver(placeStats).observe(document.body, {
      childList: true,
      subtree: true,
      characterData: true,
    });
    // The copy button copies the whole message text, so drop the stats line.
    const clipboard = navigator.clipboard;
    if (clipboard && clipboard.writeText) {
      const writeText = clipboard.writeText.bind(clipboard);
      clipboard.writeText = (text) =>
        writeText(String(text).replace(/\\n*<small class="reply-stats">[\\s\\S]*?<\\/small>\\s*$/, ""));
    }
  })();
</script>
"""


def build_demo(host: str = DEFAULT_OLLAMA_HOST) -> gr.Blocks:
    """Create the Gradio interface; ``host`` is the Ollama server address."""
    title, detail = environment_banner()
    with gr.Blocks(title=title, fill_width=True) as demo:
        history = gr.State([])  # clean messages of the current chat (sent to Ollama)
        archive = gr.State([])  # finished chats shown under "Earlier chats"
        chat_model = gr.State(DEFAULT_MODEL)  # model the current chat was started with

        with gr.Column(elem_id="app"):
            gr.Markdown(f"# {title}\n{detail}", elem_id="app-header")
            with gr.Row(elem_id="controls"):
                model = gr.Dropdown(
                    MODEL_CHOICES,
                    value=DEFAULT_MODEL,
                    label="Model",
                    filterable=False,
                    scale=3,
                )
                think = gr.Checkbox(
                    label="Thinking mode",
                    info=THINKING_INFO[supports_thinking(DEFAULT_MODEL)],
                    value=False,
                    interactive=False,
                    scale=1,
                )
            chatbot = gr.Chatbot(
                elem_id="chat",
                show_label=False,
                height="calc(100vh - 360px)",
                min_height=320,
                layout="bubble",
                buttons=["copy"],
                placeholder="### How can I help?\nPick a model above and say hello.",
            )
            box = gr.Textbox(
                placeholder="Type a message and press Enter",
                show_label=False,
                container=False,
                max_lines=6,
                submit_btn=True,
                autofocus=True,
            )

        def respond(message, hist, model_name, thinking):
            yield from chat(message, hist, model_name, thinking, host)

        def on_model_change(model_name, hist, previous_model, archive_value):
            return switch_model(model_name, hist, previous_model, archive_value, host)

        def open_chat(index, hist, model_name, archive_value):
            return open_archived_chat(index, hist, model_name, archive_value, host)

        outputs = [chatbot, history, box]
        sending = box.submit(respond, [box, history, model, think], outputs)
        model.input(
            on_model_change,
            [model, history, chat_model, archive],
            [chatbot, history, think, archive, chat_model],
            cancels=[sending],
        )

        with gr.Sidebar(width=300):
            new_chat = gr.Button("＋  Start new chat", variant="primary")
            gr.Markdown("Earlier chats", elem_classes="sidebar-heading")

            @gr.render(inputs=archive)
            def chat_list(archive_value):
                render_chat_list(
                    archive_value,
                    open_chat=open_chat,
                    open_inputs=[history, chat_model, archive],
                    open_outputs=[
                        chatbot,
                        history,
                        box,
                        archive,
                        model,
                        think,
                        chat_model,
                    ],
                    archive_state=archive,
                    cancels=[sending],
                )

        new_chat.click(
            start_new_chat,
            [history, chat_model, archive],
            [chatbot, history, box, archive],
            cancels=[sending],
        )
    return demo


def main() -> None:
    """Launch the app; configure via OLLAMA_HOST, GRADIO_SERVER_NAME, GRADIO_SERVER_PORT."""
    host = os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST
    build_demo(host).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", DEFAULT_SERVER_NAME),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", DEFAULT_SERVER_PORT)),
        theme=THEME,
        css=CSS,
        head=STATS_HEAD,
        footer_links=["gradio"],
    )


if __name__ == "__main__":
    main()
