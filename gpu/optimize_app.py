"""Runs the GEPA search inside Modal instead of on my laptop.

The reason is the API key. It lives in a Modal secret, and running the search
locally would mean exporting it into a shell, which is how I ended up rotating
a key in assignment 7. Here the secret gets injected into the container, the
optimizer talks to `hw10-serve-plain` over its own URL, and the key never
exists outside Modal.

This is CPU only. The GPU is in the other app, and all this container does is
hold the loop that sends prompts and scores what comes back.

    uv run modal run gpu/optimize_app.py
"""

import json
import pathlib

import modal

TIMEOUT_SECONDS = 3600
PROMPT_FILE = (
    pathlib.Path(__file__).resolve().parents[1] / "src" / "optimized_prompt.json"
)

# Pinned, because the whole point of this run is a before and after I can
# quote, and a different dspy could rewrite prompts differently.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("dspy==3.4.1")
    .add_local_dir("src", "/root/src")
)

app = modal.App("hw10-optimize", image=image)


@app.function(
    secrets=[modal.Secret.from_name("hw7-api-key")],
    timeout=TIMEOUT_SECONDS,
)
def run(max_metric_calls=None, hard_topic="the Millennium Falcon"):
    from src.prompt_optimization import demos_of, instructions_of, main, show

    extra = {} if max_metric_calls is None else {"max_metric_calls": max_metric_calls}
    baseline, tuned, (before, after) = main(**extra)

    show(hard_topic, baseline, "BEFORE")
    show(hard_topic, tuned, "AFTER")

    return {
        "before_instructions": instructions_of(baseline),
        "after_instructions": instructions_of(tuned),
        "demos": [
            d.toDict() if hasattr(d, "toDict") else dict(d) for d in demos_of(tuned)
        ],
        "valid_before": before,
        "valid_after": after,
    }


@app.function(
    secrets=[modal.Secret.from_name("hw7-api-key")],
    timeout=TIMEOUT_SECONDS,
)
def probe(topic="the planet Tatooine"):
    """One call against the real server, before spending a budget on a search.

    Mostly this answers whether dspy can parse what the model sends back. Its
    adapter looks for `[[ ## html ## ]]` markers and I'm asking for a document
    that starts with a doctype, so I wanted to see a real response before
    assuming the two get along.
    """
    import dspy

    from src.prompt_optimization import WritePage, build_lm, show, wait_until_ready

    wait_until_ready()
    dspy.configure(lm=build_lm())
    return show(topic, dspy.Predict(WritePage), "PROBE")


@app.function(
    secrets=[modal.Secret.from_name("hw7-api-key")],
    timeout=TIMEOUT_SECONDS,
)
def compare(topic, after_instructions):
    """Both prompts on one topic, two calls, no search.

    The search costs half an hour on a T4, and rerunning it to show one page
    would also give me a different prompt than the one I wrote up. This
    replays the saved instructions instead, so the screenshot and the write up
    are about the same run.
    """
    import dspy

    from src.prompt_optimization import WritePage, build_lm, show, wait_until_ready

    wait_until_ready()
    dspy.configure(lm=build_lm())
    show(topic, dspy.Predict(WritePage), "BEFORE")
    show(topic, dspy.Predict(WritePage.with_instructions(after_instructions)), "AFTER")


@app.local_entrypoint()
def replay(topic: str = "the Millennium Falcon"):
    saved = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
    compare.remote(topic, saved["after_instructions"])


@app.local_entrypoint()
def cli(max_metric_calls: int = 0, hard_topic: str = "the Millennium Falcon"):
    result = run.remote(
        max_metric_calls=max_metric_calls or None, hard_topic=hard_topic
    )
    PROMPT_FILE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"\nwrote {PROMPT_FILE.name}: "
        f"{result['valid_before']:.1f}% -> {result['valid_after']:.1f}% valid"
    )
