"""Serve a Gradio chat app that updates football score predictions."""

import traceback
from datetime import datetime
from threading import Lock
from zoneinfo import ZoneInfo

import modal

# Modal groups this function and its container configuration under this app.
app = modal.App("frontend-assignment")

# Install the model runtime and web UI dependencies in the container image.
image = modal.Image.debian_slim().pip_install(
    "torch",
    "transformers>=5.19.0",
    "numpy",
    "accelerate",
    "gradio",
    "fastapi",
)


def build_system_prompts(games, current_date=None):
    """Create one system prompt per game label.

    Each game tuple contains its display label and its original
    (away-team score, home-team score) prediction.
    """
    today = current_date or datetime.now(ZoneInfo("America/Denver")).date()
    return {
        game: (
            f"Today's date is {today}. You are a predictions expert tasked with updating predictions based on user's context."
            "You will be given a game, the original score prediction, and some additional context which is used to update the original prediction."
            "You will provide an updated score and explain to the user why you updated the score, including which factors played the biggest key part in updating the score."
            f"You are talking about the {game} game with the user. "
            f"The original score prediction is Away team: {prediction[0]}, "
            f"Home team: {prediction[1]}."
        )
        for game, prediction in games
    }


def load_history(histories, selected_option):
    """Return the saved conversation for one game, or an empty thread."""
    if not isinstance(histories, dict):
        histories = {}
    return histories.get(selected_option, [])


def process_message(
    message,
    histories,
    selected_option,
    system_prompts,
    generate_reply,
    reset_history=False,
):
    """Process a message and return the visible thread, cleared input, and state.

    ``generate_reply`` receives the system prompt plus valid prior turns and
    the new user message. Keeping generation injectable lets tests verify the
    resulting chatbot output without downloading or running a real model.
    """
    if not isinstance(histories, dict):
        histories = {}

    history = load_history(histories, selected_option)
    if reset_history:
        history = []

    # Keep only valid user and assistant messages in the model context.
    history = [
        {"role": item["role"], "content": item["content"]}
        for item in (history or [])
        if isinstance(item, dict)
        and item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
    ]

    if not message or not message.strip():
        # Store a reset even if the user pressed the button with an empty box.
        histories[selected_option] = history
        return history, "", histories

    messages = [
        {"role": "system", "content": system_prompts[selected_option]},
        *history,
        {"role": "user", "content": message},
    ]
    try:
        answer = generate_reply(messages)
    except (
        ImportError,
        IndexError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        # Show a useful error in the chat while preserving details in logs.
        traceback.print_exc()
        answer = f"Generation failed ({type(exc).__name__}): {exc}"

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    histories[selected_option] = history
    return history, "", histories


def generate_answer(messages, model_id, model_state, model_lock):
    """Generate text with a lazily loaded Transformers model.

    ``model_state`` stores the model and tokenizer so subsequent messages reuse
    them. ``model_lock`` prevents concurrent requests from loading or using the
    shared model at the same time.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with model_lock:
        if model_state["model"] is None:
            model_state["tokenizer"] = AutoTokenizer.from_pretrained(model_id)
            model_state["model"] = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            model_state["model"].eval()

        model = model_state["model"]
        tokenizer = model_state["tokenizer"]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=512)

        return tokenizer.decode(
            output[0][inputs.input_ids.shape[-1] :],
            skip_special_tokens=True,
        )


# Run the web app on one GPU container and keep idle containers available briefly.
@app.function(
    image=image,
    gpu="T4",
    max_containers=1,
    scaledown_window=600,
)
# Let the ASGI app serve browser requests concurrently. Model inference itself
# is serialized with model_lock so concurrent chat requests cannot race the GPU.
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def frontend():
    """Build and return the Gradio app mounted as a Modal ASGI application."""

    # Keep web-only dependencies inside the container function.
    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    # These values are shared by requests handled by this container. The model
    # is loaded on the first non-empty chat message and reused afterward.
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    model_state = {"model": None, "tokenizer": None}
    # Protect lazy model loading and generation while HTTP requests stay concurrent.
    model_lock = Lock()

    # Each tuple contains a game label and its original prediction (away score, home score).
    games = [
        ("Oregon at Michigan St. (11/20)", (31, 15)),
        ("Iowa at Illinois (11/21)", (24, 16)),
        ("Arkansas at Texas (11/21)", (14, 34)),
        ("Utah at TCU (11/21)", (27, 20)),
        ("BYU at Kansas (11/21)", (27, 21)),
        ("Wake Forest at Georgia Tech (11/21)", (38, 18)),
        ("Wofford at Ole Miss (11/21)", (13, 53)),
        ("Ohio St. at Nebraska (11/21)", (22, 19)),
        ("Houston at West Virginia (11/21)", (39, 31)),
        ("UCLA at Michigan (11/21)", (21, 17)),
        ("LSU at Tennessee (11/21)", (24, 22)),
    ]
    # Use each game label as both the sidebar value and its prompt/history key.
    system_prompts = build_system_prompts(games)

    def respond(message, history, selected_option, reset_history=False):
        """Connect the Gradio callback to the reusable response helpers."""
        return process_message(
            message,
            history,
            selected_option,
            system_prompts,
            lambda messages: generate_answer(
                messages, model_id, model_state, model_lock
            ),
            reset_history,
        )

    # Define the sidebar, chat transcript, input box, and fresh-hypothesis action.
    with gr.Blocks() as chat:
        gr.Markdown("## Qwen2.5 Chat")
        # Store all game threads together so switching games preserves each one.
        chat_history = gr.State({option: [] for option in system_prompts})
        with gr.Row():
            with gr.Column(scale=1, min_width=220):
                gr.Markdown("### Game")
                selected_option = gr.Radio(
                    choices=list(system_prompts),
                    value=games[0][0],
                    label="Choose a game",
                )
            with gr.Column(scale=4):
                chatbot = gr.Chatbot()
                message = gr.Textbox(placeholder="Send a message", show_label=False)
                test_hypothesis = gr.Button("Test New Hypothesis")

        # Pressing Enter submits with existing context for the selected game.
        message.submit(
            respond,
            inputs=[message, chat_history, selected_option],
            outputs=[chatbot, message, chat_history],
            queue=False,
        )
        # Starting a new hypothesis clears the selected game's thread before
        # using the current textbox message to begin a new one.
        test_hypothesis.click(
            lambda message, history, option: respond(
                message, history, option, reset_history=True
            ),
            inputs=[message, chat_history, selected_option],
            outputs=[chatbot, message, chat_history],
            queue=False,
        )
        # Switching games changes the visible transcript without clearing it.
        selected_option.change(
            load_history,
            inputs=[chat_history, selected_option],
            outputs=[chatbot],
            queue=False,
        )

    # Expose Gradio through FastAPI at the root URL of the Modal web function.
    return mount_gradio_app(FastAPI(), chat, path="/")
