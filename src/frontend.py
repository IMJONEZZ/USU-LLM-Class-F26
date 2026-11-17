"""A browser front end for the two HTML services, with a switch between them.

Assignment 9 put a grammar on the server so invalid markup couldn't be
sampled. Assignment 10 took the grammar off and optimized the prompt instead.
I had both numbers and no way to show anybody the difference, so this puts
them behind one text box and a radio button.

Both panels matter and neither is enough on its own. The rendered page is what
makes this feel like a product, but browsers repair broken HTML silently, so a
page with four unclosed tags can draw perfectly and look like a success. The
raw markup is the only place that shows. The verdict line underneath comes
from the same `validate()` the last two assignments were scored with.

The service callables are arguments. The tests pass fakes, which is how this
file is covered without a GPU, a key or a network.

    uv run python -m src.frontend
"""

import html
import json
import pathlib
from functools import partial

import requests

from src.html_output import validate
from src.inference import ServerNotReady

MODE_GRAMMAR = "Grammar on (assignment 9)"
MODE_PROMPT = "Optimized prompt (assignment 10)"
MODES = (MODE_GRAMMAR, MODE_PROMPT)

PROMPT_FILE = pathlib.Path(__file__).resolve().parent / "optimized_prompt.json"

TITLE = "HTML generation: grammar against optimized prompt"
DESCRIPTION = (
    "Same model, same quant, same T4. The only thing the switch changes is "
    "whether a GBNF grammar is loaded on the server. Pick a topic, or use "
    "Compare both to send one topic to both services at once."
)

# Two sets on purpose. The spread is there to show the thing works on anything,
# and the Star Wars ones are what I've been testing against all semester, so
# they're the ones where I already know what good output looks like.
PLAIN_EXAMPLES = (
    "how tides work",
    "the Rosetta Stone",
    "why onions make you cry",
    "the Antikythera mechanism",
)
STAR_WARS_EXAMPLES = (
    "the Millennium Falcon",
    "the planet Tatooine",
    "the Battle of Hoth",
    "Jabba's palace",
)
EXAMPLES = PLAIN_EXAMPLES + STAR_WARS_EXAMPLES

# Two tries, because the first is what wakes the container and the second
# is the one that gets an answer.
COLD_ATTEMPTS = 2

WAKING_UP = (
    "The server is waking up. It scales to zero, so the first request after "
    "an idle spell waits on a 4.7GB model loading onto a GPU."
)


def optimized_instructions(path=None):
    """The instruction GEPA wrote in assignment 10, read off disk.

    Committed rather than regenerated, because GEPA is stochastic and a fresh
    search would give a different prompt than the one I measured.
    """
    saved = json.loads(pathlib.Path(path or PROMPT_FILE).read_text(encoding="utf-8"))
    return saved["after_instructions"]


def grammar_service(topic, **kwargs):
    from src.html_output import generate

    return generate(topic, **kwargs)["html"]


def prompt_service(topic, instructions=None, **kwargs):
    """Assignment 10's path: no grammar, the optimized instruction does the work.

    This goes through dspy on purpose. Hand rolling the prompt would almost
    certainly format it differently than the optimizer did, and then the page
    on screen wouldn't be the thing I measured.
    """
    import dspy

    from src.prompt_optimization import WritePage, build_lm, wait_until_ready

    wait_until_ready()
    dspy.configure(lm=build_lm(**kwargs))
    signature = WritePage.with_instructions(instructions or optimized_instructions())
    return dspy.Predict(signature)(topic=topic).html


SERVICES = {MODE_GRAMMAR: grammar_service, MODE_PROMPT: prompt_service}


def verdict(problems):
    """One line a person can read, from the validator's list of complaints."""
    if not problems:
        return "Valid. Nothing outside the grammar."
    head = "; ".join(problems[:3])
    more = f", and {len(problems) - 3} more" if len(problems) > 3 else ""
    return f"Invalid, {len(problems)} problems. {head}{more}"


def answer(topic, mode, services=None, attempts=COLD_ATTEMPTS):
    """Run one topic through one service and score what comes back.

    Nothing raises out of here. An exception that escapes lands in the browser
    as a red Gradio popup with a stack trace behind it, which is useless to
    anyone looking at the page, and a cold server is a normal thing to hit
    rather than a crash. Both come back as text in the verdict line instead.

    The retry is for a cold GPU specifically. `ask` waits out the 503 that
    llama.cpp returns while it loads, but a server that's properly asleep
    holds the connection open instead and `requests` gives up at its own
    timeout, which is not a 503 and never reached that retry loop. The first
    attempt is what wakes the container, so the second one usually lands. I
    found this on the last check before calling the assignment done, which is
    the one path a grader opening a cold URL would have hit first.
    """
    services = SERVICES if services is None else services
    for attempt in range(1, attempts + 1):
        try:
            markup = services[mode](topic)
        except ServerNotReady, requests.Timeout, requests.ConnectionError:
            if attempt == attempts:
                return {"mode": mode, "markup": "", "problems": [], "note": WAKING_UP}
        # Broad on purpose. This is the edge of the app, and anything that
        # gets past here is a popup with a stack trace in a browser.
        except Exception as exc:  # noqa: BLE001
            note = f"{mode} didn't answer. {type(exc).__name__}: {exc}"
            return {"mode": mode, "markup": "", "problems": [], "note": note}
        else:
            return {
                "mode": mode,
                "markup": markup,
                "problems": validate(markup),
                "note": None,
            }
    raise AssertionError("unreachable")  # pragma: no cover


def panels(result):
    """The three things the UI shows: rendered page, raw markup, verdict.

    The verdict gets escaped on the way out. Gradio renders chat messages as
    markdown, and every complaint the validator makes names a tag, so an
    unescaped "<strong> is not a tag the grammar allows" renders as a bolded
    "is not a tag the grammar allows" with the tag swallowed and the rest of
    the line in bold. The one word worth reading is the one that disappears.
    """
    line = result.get("note") or verdict(result["problems"])
    return result["markup"], result["markup"], html.escape(line)


def respond(message, history, mode, services=None):
    """What ChatInterface calls. Returns the chat line plus the side panels.

    The topic comes back as the last value so it can fill the compare box.
    ChatInterface clears its own textbox on submit, so without this the
    compare button would have nothing left to read. That box is a visible
    Textbox and not a `gr.State` on purpose: state doesn't serialize over the
    API, so a hidden one is a feature I could only check by clicking it.
    """
    topic = (message or "").strip()
    if not topic:
        return "Give me a topic first.", "", "", ""
    result = answer(topic, mode, services)
    rendered, raw, line = panels(result)
    return f"{mode}: {line}", rendered, raw, topic


def compare(topic, services=None):
    """Both services on one topic, for the side by side panels."""
    topic = (topic or "").strip()
    if not topic:
        empty = ("", "", "Give me a topic first.")
        return empty + empty
    left = panels(answer(topic, MODE_GRAMMAR, services))
    right = panels(answer(topic, MODE_PROMPT, services))
    return left + right


def build_app(services=None):
    import gradio as gr

    # `partial` rather than a closure so the injected services are bound
    # without adding a wrapper body, and `api_name` below is what names the
    # endpoints. Left to itself gradio called them /lambda_2 and /lambda_15.
    on_message = partial(respond, services=services)
    on_compare = partial(compare, services=services)

    with gr.Blocks(title=TITLE) as app:
        gr.Markdown(f"# {TITLE}\n\n{DESCRIPTION}")

        with gr.Row():
            rendered = gr.HTML(label="Rendered")
            raw = gr.Code(label="Raw markup", language="html")

        mode = gr.Radio(list(MODES), value=MODE_GRAMMAR, label="Service")
        topic_box = gr.Textbox(
            label="Topic for Compare both",
            placeholder="fills in from the chat, or type one here",
        )

        gr.ChatInterface(
            fn=on_message,
            api_name="chat",
            chatbot=gr.Chatbot(height=300, label="Chatbot"),
            additional_inputs=[mode],
            additional_outputs=[rendered, raw, topic_box],
            # Gradio wants a row per example once there are additional inputs,
            # so each one has to carry a mode as well as a topic. They all say
            # grammar, because an example that quietly flipped the switch
            # underneath you would be a mean thing to build.
            examples=[[topic, MODE_GRAMMAR] for topic in EXAMPLES],
            title=None,
            description=None,
        )

        gr.Markdown("### Both at once")
        compare_button = gr.Button("Compare both on the last topic")
        with gr.Row():
            with gr.Column():
                gr.Markdown(f"**{MODE_GRAMMAR}**")
                left_rendered = gr.HTML()
                left_raw = gr.Code(language="html")
                left_checked = gr.Markdown("")
            with gr.Column():
                gr.Markdown(f"**{MODE_PROMPT}**")
                right_rendered = gr.HTML()
                right_raw = gr.Code(language="html")
                right_checked = gr.Markdown("")

        # The order here has to match what `compare` returns, and nothing
        # checks that for me, so there's a test that does.
        compare_button.click(
            fn=on_compare,
            api_name="compare",
            inputs=[topic_box],
            outputs=[
                left_rendered,
                left_raw,
                left_checked,
                right_rendered,
                right_raw,
                right_checked,
            ],
        )
    return app


if __name__ == "__main__":  # pragma: no cover - entry point, not logic
    build_app().launch()
