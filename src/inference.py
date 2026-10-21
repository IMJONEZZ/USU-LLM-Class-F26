"""Send a real HTTP request to the Assignment 7 FastAPI model server."""

# --- Imports ---
# argparse: turn command-line flags into Python values (e.g. --url, the prompt).
# json: pretty-print dictionaries as JSON text.
# sys: write errors to stderr and exit with a status code.
# requests: HTTP client library (like curl, but from Python).
import argparse
import json
import sys

import requests


def main(argv=None):
    """Parse CLI args, POST to /generate, print the reply, return 0=ok or 1=fail.

    argv=None means "use whatever was typed on the command line".
    Tests pass a fake argv list so they don't need a real terminal.
    """
    # Build a CLI help/parser. __doc__ is this file's top triple-quoted string.
    parser = argparse.ArgumentParser(description=__doc__)

    # Positional prompt: optional (nargs="?"). If omitted, use the default question.
    # Example: python -m src.inference "What is PEFT?"
    parser.add_argument(
        "prompt", nargs="?", default="How many Ls are in the word parallel?"
    )
    # Flags start with --. Defaults match the local server from `make serve*`.
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300)

    # Convert argv into an object: args.prompt, args.url, etc.
    args = parser.parse_args(argv)

    # Validate early so we fail with a clear message before calling the server.
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not 1 <= args.max_new_tokens <= 256:
        parser.error("--max-new-tokens must be between 1 and 256")
    if not args.prompt.strip():
        parser.error("prompt must not be blank")

    # Server endpoint is always .../generate. rstrip("/") avoids //generate.
    url = args.url.rstrip("/") + "/generate"
    # Body the API expects (same shape as in /docs).
    payload = {"prompt": args.prompt, "max_new_tokens": args.max_new_tokens}

    # flush=True prints immediately (useful while waiting on a slow model).
    print(f"POST {url}", flush=True)
    print(json.dumps(payload, indent=2), flush=True)

    try:
        # POST with JSON body. timeout is how long we wait for a reply (seconds).
        response = requests.post(url, json=payload, timeout=args.timeout)
        print(f"HTTP {response.status_code}")
        # raise_for_status() turns 4xx/5xx into an exception.
        response.raise_for_status()
        # response.json() parses the JSON body into a Python dict, then we print it.
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except (requests.RequestException, ValueError) as exc:
        # Network errors, timeouts, bad HTTP status, or invalid JSON land here.
        print(f"Request failed: {exc}", file=sys.stderr)
        # If the server sent an error body, show it too.
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            print(exc.response.text, file=sys.stderr)
        return 1  # nonzero = failure (shell / make will treat this as an error)
    return 0  # success


# Only run main() when this file is executed as a script:
#   uv run python -m src.inference
# Importing the module from tests does NOT run main() automatically.
if __name__ == "__main__":
    # SystemExit(code) exits the process with that status code.
    raise SystemExit(main())
