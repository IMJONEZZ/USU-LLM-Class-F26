"""Client for the model I'm serving in assignment 7.

The server is llama.cpp behind a Modal URL (`gpu/serve_app.py`) and it speaks
the OpenAI chat completions API, so this is an ordinary POST with a bearer
token. Nothing here knows anything about llama.cpp specifically, which is the
point of having picked an OpenAI compatible server.

The server scales to zero, so the first call after it's been idle arrives while
llama.cpp is still reading an 800MB file off a volume. That comes back as a 503
with `"Loading model"` in it, not as a connection error, and the right response
is to wait and ask again, not to give up.

The session is an argument. The tests pass a fake one, which is how they cover
this file without a server to talk to and without the CI runner needing network
access at all.

    export LLAMA_API_KEY=...
    uv run python -m src.inference "How many Ls are in the word parallel?"
"""

import json
import os
import time

import requests

BASE_URL = "https://froostysnoowman--hw7-serve-serve.us-east.modal.run"
MODEL = "hw6-trick-questions"

# Cold start is about ten seconds in practice. Six tries at five seconds gives
# half a minute, which is enough slack for a cold container without hanging
# forever if the server is genuinely broken.
MAX_ATTEMPTS = 6
RETRY_SECONDS = 5.0
TIMEOUT_SECONDS = 120


class ServerNotReady(RuntimeError):
    """Raised when the server is still loading after every retry."""


def api_key(env=None):
    """Read the key from the environment.

    Keeping it out of this file is the whole point. The repo is public and
    the server is a public URL, so a key committed here would be worth
    exactly nothing the moment it landed.
    """
    env = os.environ if env is None else env
    key = env.get("LLAMA_API_KEY")
    if not key:
        raise RuntimeError(
            "LLAMA_API_KEY is not set. The server rejects unauthenticated "
            "requests, so there's nothing useful to do without it."
        )
    return key


def build_request(question, model=MODEL, max_tokens=200, temperature=0.0):
    """Build the JSON body for one chat completion.

    Temperature is zero so the same question gives the same answer. The whole
    reason I can compare this against the assignment 6 numbers is that neither
    side is sampling.
    """
    return {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def is_still_loading(response):
    """True when the server is up but hasn't finished loading the model.

    llama.cpp answers 503 with an `unavailable_error` while it reads the GGUF.
    A 503 from anything else in front of it means something actually broke, so
    this looks at the body rather than trusting the status code alone.
    """
    if response.status_code != 503:
        return False
    try:
        body = response.json()
    except ValueError:
        return True
    message = str(body.get("error", {}).get("message", "")).lower()
    return "loading" in message or not message


def ask(
    question,
    url=BASE_URL,
    key=None,
    session=None,
    attempts=MAX_ATTEMPTS,
    retry_seconds=RETRY_SECONDS,
    sleep=time.sleep,
    **kwargs,
):
    """Ask one question and return the answer text.

    `session` and `sleep` are arguments so the tests can hand over a fake
    transport and a clock that doesn't actually wait.
    """
    session = requests.Session() if session is None else session
    headers = {
        "Authorization": f"Bearer {key or api_key()}",
        "Content-Type": "application/json",
    }
    payload = build_request(question, **kwargs)

    for attempt in range(1, attempts + 1):
        response = session.post(
            f"{url}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )
        if is_still_loading(response):
            if attempt == attempts:
                break
            print(f"  server still loading, retry {attempt}/{attempts - 1}")
            sleep(retry_seconds)
            continue
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    raise ServerNotReady(
        f"server was still loading after {attempts} attempts over "
        f"{attempts * retry_seconds:.0f}s"
    )


def show(question, **kwargs):
    """Print the request and the response the way the homework asks for."""
    payload = build_request(
        question,
        **{
            k: v
            for k, v in kwargs.items()
            if k in ("model", "max_tokens", "temperature")
        },
    )
    print("REQUEST")
    print(f"  POST {kwargs.get('url', BASE_URL)}/v1/chat/completions")
    # Redacted. The key is in the PR comment because whoever grades this
    # needs it, but a screenshot is forever and that's a different thing.
    print("  Authorization: Bearer ****")
    print("  Content-Type: application/json")
    for line in json.dumps(payload, indent=2).splitlines():
        print(f"  {line}")

    started = time.time()
    answer = ask(question, **kwargs)
    print(f"\nRESPONSE  ({time.time() - started:.1f}s)")
    for line in answer.splitlines():
        print(f"  {line}")
    return answer


if __name__ == "__main__":  # pragma: no cover - entry point, not logic
    import sys

    asked = (
        sys.argv[1] if len(sys.argv) > 1 else "How many Ls are in the word parallel?"
    )
    show(asked)
