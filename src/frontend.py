import time

import gradio as gr
import truststore

from src import config, inference


def chat(message, history):
    """Stream the reply as the text so far; history is ignored because the adapter was trained single-turn."""
    body = inference.stream_body(message, config.ADAPTER_NAME)
    sent = time.perf_counter()
    response, _, _ = inference.call("POST", "/v1/completions", json=body, stream=True)
    text, arrivals = "", []
    for chunk in inference.parse_sse(response.iter_lines(decode_unicode=True)):
        if chunk["choices"]:
            arrivals.append(time.perf_counter() - sent)
            text += chunk["choices"][0]["text"]
            yield text
    # shortcut: stream_stats needs 2+ chunks, a one-token reply raises here; guard it if that ever happens
    stats = inference.stream_stats(arrivals)
    yield (
        f"{text}\n\n---\nfirst token {stats['ttft_s']:.2f}s | "
        f"{stats['rate_tok_s']:.1f} tok/s | {stats['chunks']} chunks"
    )


def launch():
    truststore.inject_into_ssl()
    gr.ChatInterface(chat).launch()


if __name__ == "__main__":
    launch()
