"""Send a prompt to the deployed Star Wars model using the OpenAI SDK."""

import argparse
import os

from openai import OpenAI

DEFAULT_MODEL = "star-wars"
DEFAULT_BASE_URL = (
    "https://cameronswapp--star-wars-lora-serving-serve.us-east.modal.run/v1"
)


def run_inference(prompt: str, base_url: str, model: str = DEFAULT_MODEL) -> str:
    """Send one prompt to an OpenAI-compatible API and return the response text."""
    client = OpenAI(
        base_url=base_url.rstrip("/"),
        # vLLM doesn't require an API key unless authentication is configured.
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )
    response = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=128,
    )
    return response.choices[0].text or ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="Prompt to send to the model")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("MODAL_VLLM_BASE_URL", DEFAULT_BASE_URL),
        help="Modal endpoint URL including /v1",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    print(run_inference(args.prompt, args.base_url, model=args.model))


if __name__ == "__main__":  # pragma: no cover
    main()
