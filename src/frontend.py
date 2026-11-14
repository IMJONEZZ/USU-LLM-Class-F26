"""
A simple Gradio chat frontend powered by Groq's hosted API, themed as a
Star Wars assistant via a system prompt. The actual response-generation
logic lives in a plain function (get_response), separate from Gradio's
ChatInterface wiring, so it can be tested independently without needing a
real API key or network access in CI.
"""

import os

import gradio as gr
from groq import Groq

SYSTEM_PROMPT = (
    "You are a knowledgeable Star Wars assistant, speaking with the calm, wise tone of a Jedi "
    "Master. Feel free to reference Star Wars lore, characters, and quotes when relevant, even "
    "for questions not directly about Star Wars."
)


def get_response(message, history, client=None):
    """
    Given a user message and conversation history, return the model's
    response. The Groq client is injected as a parameter (defaulting to a
    real client) so tests can supply a fake client instead of making a
    real API call.
    """
    if client is None:
        client = Groq(api_key=os.environ["GROQ_API_KEY"])  # pragma: no cover

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
    )
    return response.choices[0].message.content


def build_interface():  # pragma: no cover
    """Build and return the Gradio ChatInterface."""
    return gr.ChatInterface(
        fn=get_response,
        title="Star Wars Assistant",
    )


if __name__ == "__main__":  # pragma: no cover
    demo = build_interface()
    demo.launch()
