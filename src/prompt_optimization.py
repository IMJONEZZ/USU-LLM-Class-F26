"""Optimizing a prompt until it produces the HTML the grammar used to force.

Assignment 9 put a GBNF grammar on the server, so invalid markup wasn't
something the sampler could reach. This asks the opposite question. With the
grammar off, how close can a prompt get on its own, and what does the gap look
like? The scorer is `validate()` from `src/html_output.py`, which is the same
check the last assignment used, so a page that would have passed there passes
here.

GEPA is the optimizer because the scorer talks. `validate()` hands back lines
like "<div> is not a tag the grammar allows", and GEPA's metric returns written
feedback instead of a number, so those lines go straight to the model that
rewrites the instructions. Sending it a 0 or a 1 would throw that away.

The LM is an argument everywhere, which is how the tests cover this file
without a GPU, a 4.7GB download or a network.

    export LLAMA_API_KEY=...
    uv run python -m src.prompt_optimization
"""

import re

import dspy

from src.html_output import (
    COLD_START_ATTEMPTS,
    COLD_START_SECONDS,
    validate,
)
from src.inference import api_key, ask

BASE_URL = "https://froostysnoowman--hw10-serve-plain-serve.us-east.modal.run"

# Two spellings of the same thing. llama-server was started with this alias,
# and LiteLLM needs the `openai/` prefix to know which API to speak.
SERVER_MODEL = "qwen2.5-7b"
MODEL = f"openai/{SERVER_MODEL}"

# The budget. GEPA counts metric calls, so this is the knob that decides what
# the run costs, and a fixed number is easier to reason about than a preset.
MAX_METRIC_CALLS = 240
REFLECTION_MINIBATCH = 3

# A page that validates and says nothing is the cheapest way to score well, so
# validity on its own isn't enough. These are the floor: the page has to be
# about the topic it was given and long enough to be worth reading.
MIN_BLOCKS = 3
MIN_TEXT_CHARS = 200

# Where valid markup starts. Anything that fails the validator scores below
# this, anything that passes scores at least this, so "it parses" is never a
# trade against "it reads nicely".
VALID_SCORE = 0.5
CONTENT_CHECKS = 3

TOPICS = (
    # Train. Deliberately a mix, because left to myself I'd write forty Star
    # Wars topics and then be optimizing for one narrow kind of page.
    "the planet Tatooine",
    "how sourdough starter works",
    "the rules of cricket",
    "the Voyager 2 mission",
    "lighthouses",
    "why onions make you cry",
    "the Mariana Trench",
    "Ada Lovelace",
    "how a bicycle derailleur works",
    "the city of Petra",
    "octopus intelligence",
    "the Dewey Decimal System",
    # Dev.
    "the Battle of Hoth",
    "tectonic plates",
    "the history of the fork",
    "how noise cancelling headphones work",
    "Icelandic turf houses",
    "the rules of chess",
    "bioluminescence",
    "the Silk Road",
    # Test. Bigger than the train set, and the optimizer never sees it.
    "the Millennium Falcon",
    "how vaccines work",
    "the Great Barrier Reef",
    "Morse code",
    "the Trans-Siberian Railway",
    "sleep paralysis",
    "the invention of the printing press",
    "how tides work",
    "Komodo dragons",
    "the Rosetta Stone",
    "why the sky is blue",
    "the Globe Theatre",
    "coffee processing methods",
    "the Antikythera mechanism",
    "how elevators are counterweighted",
    "the aurora borealis",
    "paper money",
    "the deepest cave in the world",
    "why cats purr",
    "the Suez Canal",
)
SPLITS = {"train": slice(0, 12), "dev": slice(12, 20), "test": slice(20, None)}

BLOCK_TAGS = ("h1", "h2", "p", "li")


class WritePage(dspy.Signature):
    """Write an HTML page about the topic."""

    topic: str = dspy.InputField()
    html: str = dspy.OutputField()


def examples(split):
    """Topics as dspy Examples. There's no gold page, only an input."""
    return [dspy.Example(topic=t).with_inputs("topic") for t in TOPICS[SPLITS[split]]]


def visible_text(markup):
    return re.sub(r"<[^>]*>", " ", markup)


def content_problems(topic, markup):
    """What's wrong with the page as writing, assuming it's valid markup.

    Without this the optimizer can satisfy the scorer with four tags and no
    words, and since it only ever sees the score, that's the cheapest page it
    can learn to write.
    """
    problems = []
    blocks = sum(len(re.findall(rf"<{tag}>", markup)) for tag in BLOCK_TAGS)
    if blocks < MIN_BLOCKS:
        problems.append(
            f"only {blocks} blocks of content, it needs at least {MIN_BLOCKS}"
        )

    text = visible_text(markup)
    if len(text.strip()) < MIN_TEXT_CHARS:
        problems.append(
            f"only {len(text.strip())} characters of text, it reads as a stub"
        )

    title = re.search(r"<title>(.*?)</title>", markup, re.DOTALL)
    words = {w for w in re.findall(r"[a-z]{4,}", topic.lower())}
    if words and title and not any(w in title.group(1).lower() for w in words):
        problems.append(f"the title {title.group(1)!r} doesn't mention the topic")
    return problems


def score_page(topic, markup):
    """Return a score in [0, 1] and the problems behind it.

    Valid markup is always worth more than invalid markup, so everything that
    fails `validate()` lives below VALID_SCORE and everything that passes
    lives at or above it. Inside each band the score is graded.

    The grading is the whole reason this isn't a pass or fail. My first
    version returned 0.0 for anything with a structural problem, and GEPA came
    back from a full run with the prompt byte for byte unchanged, because
    every candidate it tried scored exactly zero and there was nothing to tell
    "almost valid" from "nothing like HTML". A page with one stray tag out of
    twenty now outscores one that's twenty for twenty wrong, which is the
    difference between a search and a coin flip.
    """
    structural = validate(markup)
    if structural:
        tags = len(re.findall(r"<[a-zA-Z/][^>]*>", markup))
        if not tags:
            return 0.0, structural
        # Problems over problems plus tags, not problems over tags. One bad
        # <div> earns three complaints (wrong tag, has attributes, never
        # closed), so the plain ratio saturates at 1 and every broken page
        # lands on zero together, which is the flat start I'm trying to avoid.
        share_wrong = len(structural) / (len(structural) + tags)
        return VALID_SCORE * (1.0 - share_wrong), structural

    content = content_problems(topic, markup)
    if not content:
        return 1.0, []
    kept = 1.0 - len(content) / CONTENT_CHECKS
    return VALID_SCORE + (1.0 - VALID_SCORE) * kept, content


def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """GEPA's metric: a score plus something the reflection model can read.

    The feedback is only what went wrong with this page. My first version
    appended the list of allowed tags, and then GEPA scored 100% off a budget
    of twenty calls, because it wasn't searching for anything, it was copying
    a spec I'd handed it. What's left is what a validator actually knows, so
    the rules have to be inferred from which pages get complained about.
    """
    markup = getattr(prediction, "html", "") or ""
    value, problems = score_page(example.topic, markup)
    if not problems:
        return dspy.Prediction(
            score=value, feedback="Valid HTML with real content in it."
        )
    feedback = f"Scored {value}. Problems with this page: " + "; ".join(problems) + "."
    return dspy.Prediction(score=value, feedback=feedback)


def accuracy(example, prediction, trace=None):
    value, _ = score_page(example.topic, getattr(prediction, "html", "") or "")
    return value


def wait_until_ready(url=BASE_URL, key=None, **kwargs):
    """Block until the container has the model loaded.

    llama.cpp answers 503 with "Loading model" while it reads 4.7GB off the
    volume, and dspy treats that as a server error and stops. My own client
    from assignment 7 already knows how to wait it out, so the cold start gets
    absorbed here and dspy only ever meets a server that's up.
    """
    return ask(
        "ready?",
        url=url,
        key=key,
        model=SERVER_MODEL,
        max_tokens=1,
        attempts=kwargs.pop("attempts", COLD_START_ATTEMPTS),
        retry_seconds=kwargs.pop("retry_seconds", COLD_START_SECONDS),
        **kwargs,
    )


def build_lm(url=BASE_URL, key=None, model=MODEL, temperature=0.0, **kwargs):
    return dspy.LM(
        model,
        api_base=f"{url}/v1",
        api_key=key or api_key(),
        temperature=temperature,
        cache=False,
        **kwargs,
    )


def predictor(program):
    """ChainOfThought keeps its predictor on `.predict`, Predict is its own."""
    return getattr(program, "predict", program)


def instructions_of(program):
    return predictor(program).signature.instructions


def demos_of(program):
    return list(getattr(predictor(program), "demos", []))


def optimize(
    program, lm, trainset, valset, reflection_lm=None, optimizer=None, **kwargs
):
    """Run GEPA and hand back the rewritten program.

    `optimizer` is injected so the tests can stand something cheap in its
    place. Running the real search in CI would need a GPU and about twenty
    minutes, and what I actually want to know is that my wiring is right.
    """
    maker = optimizer or dspy.GEPA
    search = maker(
        metric=metric,
        reflection_lm=reflection_lm or lm,
        max_metric_calls=kwargs.pop("max_metric_calls", MAX_METRIC_CALLS),
        reflection_minibatch_size=kwargs.pop(
            "reflection_minibatch_size", REFLECTION_MINIBATCH
        ),
        **kwargs,
    )
    return search.compile(program, trainset=trainset, valset=valset)


def measure(program, devset, num_threads=4):
    """Generate once, report both numbers.

    The obvious way to get a validity rate and a mean score is two calls to
    `dspy.Evaluate`, and that generates every page twice. The scores come back
    attached to the predictions, so the second number is arithmetic on pages I
    already paid for.
    """
    scorer = dspy.Evaluate(
        devset=devset, metric=accuracy, num_threads=num_threads, display_progress=False
    )
    result = scorer(program)
    pages = [
        getattr(prediction, "html", "") or "" for _, prediction, _ in result.results
    ]
    valid = sum(1 for page in pages if not validate(page))
    return 100.0 * valid / len(pages), result.score, pages


def main(lm=None, reflection_lm=None, optimizer=None, num_threads=4, **kwargs):
    """Measure the stock prompt, optimize it, measure again on unseen topics."""
    if lm is None:
        wait_until_ready()
        lm = build_lm()
    dspy.configure(lm=lm)

    train, dev, test = examples("train"), examples("dev"), examples("test")
    baseline = dspy.Predict(WritePage)

    before_valid, before_score, _ = measure(baseline, test, num_threads)

    tuned = optimize(
        baseline,
        lm,
        train,
        dev,
        reflection_lm=reflection_lm,
        optimizer=optimizer,
        **kwargs,
    )

    after_valid, after_score, _ = measure(tuned, test, num_threads)

    print("\nPROMPT BEFORE")
    print(f"  {instructions_of(baseline)}")
    print(f"  demos: {len(demos_of(baseline))}")
    print("\nPROMPT AFTER")
    for line in instructions_of(tuned).splitlines():
        print(f"  {line}")
    print(f"  demos: {len(demos_of(tuned))}")
    print(
        f"\nValid HTML on {len(test)} unseen topics: "
        f"{before_valid:.1f}% -> {after_valid:.1f}%"
    )
    print(
        f"Mean score out of 100 (validity plus content floor): "
        f"{before_score:.1f} -> {after_score:.1f}"
    )
    return baseline, tuned, (before_valid, after_valid)


def show(topic, program, label):
    """Print one page and whether it passed, which is what the screenshot is."""
    prediction = program(topic=topic)
    markup = getattr(prediction, "html", "") or ""
    print(f"\n=== {label} === {topic}")
    for line in markup.splitlines():
        print(f"  {line}")
    value, problems = score_page(topic, markup)
    print(f"  CHECK  score {value}")
    for problem in problems or ["nothing wrong with it"]:
        print(f"    {problem}")
    return markup


if __name__ == "__main__":  # pragma: no cover - entry point, not logic
    main()
