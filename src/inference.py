"""Client for the Assignment 7 vLLM server (OpenAI-compatible chat API)."""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

import requests

DEFAULT_BASE_URL = "https://p-vavilala--assignment-7-vllm-serve.us-east.modal.run"
DEFAULT_MODEL = "a6"
API_KEY_ENV = "VLLM_API_KEY"
BASE_URL_ENV = "A7_SERVER_URL"
# A cold start took about 200 seconds in my first measurement, so a request that
# wakes the server must be allowed to wait longer than that.
DEFAULT_TIMEOUT_SECONDS = 400.0
DEFAULT_RETRIES = 2
RETRY_DELAY_SECONDS = 2.0


class InferenceError(RuntimeError):
    """Raised when the server cannot give a usable answer."""


def build_headers(api_key: str) -> dict[str, str]:
    """Return the headers the protected server expects."""
    if not api_key:
        raise InferenceError(f"No API key given; set {API_KEY_ENV}.")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def build_payload(
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Build a greedy chat request, matching the Assignment 6 answer settings."""
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def _send(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    timeout: float,
    retries: int,
    json: dict[str, Any] | None = None,
) -> requests.Response:
    """Send a request, retrying only timeouts, connection errors and 5xx replies."""
    last_error = "no attempt made"
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(RETRY_DELAY_SECONDS)
        try:
            response = requests.request(
                method, url, headers=headers, json=json, timeout=timeout
            )
        except (requests.Timeout, requests.ConnectionError) as error:
            last_error = f"{type(error).__name__}: {error}"
            continue
        if response.status_code >= 500:
            last_error = f"HTTP {response.status_code}"
            continue
        if not response.ok:
            # A client error such as a wrong key will not fix itself on retry.
            raise InferenceError(f"HTTP {response.status_code}: {response.text[:200]}")
        return response
    raise InferenceError(f"Gave up after {retries + 1} attempts ({last_error}).")


def chat(
    messages: list[dict[str, str]],
    *,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
) -> dict[str, Any]:
    """Send one chat request and return the answer text, usage and latency."""
    started = time.perf_counter()
    response = _send(
        "POST",
        f"{base_url.rstrip('/')}/v1/chat/completions",
        headers=build_headers(api_key),
        json=build_payload(model, messages, max_tokens, temperature),
        timeout=timeout,
        retries=retries,
    )
    latency = time.perf_counter() - started
    try:
        body = response.json()
        text = body["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise InferenceError(f"Unexpected reply from server: {error!r}") from error
    return {
        "model": model,
        "text": text,
        "usage": body.get("usage", {}),
        "latency_seconds": round(latency, 3),
    }


def ask(question: str, **kwargs: Any) -> dict[str, Any]:
    """Ask one plain question with no system prompt."""
    return chat([{"role": "user", "content": question}], **kwargs)


def list_models(
    *,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
) -> list[str]:
    """Return the model names the server offers (the base model and adapters)."""
    response = _send(
        "GET",
        f"{base_url.rstrip('/')}/v1/models",
        headers=build_headers(api_key),
        timeout=timeout,
        retries=retries,
    )
    try:
        return [item["id"] for item in response.json()["data"]]
    except (ValueError, KeyError, TypeError) as error:
        raise InferenceError(f"Unexpected reply from server: {error!r}") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the Assignment 7 server.")
    parser.add_argument("question", help="the question to send")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="a6 or the base model")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args(argv)

    api_key = os.environ.get(API_KEY_ENV, "")
    base_url = os.environ.get(BASE_URL_ENV, DEFAULT_BASE_URL)
    print(f"POST {base_url}/v1/chat/completions")
    print("Authorization: Bearer ****")  # never print the real key
    print(f"model={args.model} max_tokens={args.max_tokens} temperature=0")
    print(f"question: {args.question}")
    try:
        result = ask(
            args.question,
            api_key=api_key,
            base_url=base_url,
            model=args.model,
            max_tokens=args.max_tokens,
        )
    except InferenceError as error:
        print(f"error: {error}")
        return 1
    print(f"answer: {result['text']}")
    print(f"usage: {result['usage']}  latency: {result['latency_seconds']} s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
