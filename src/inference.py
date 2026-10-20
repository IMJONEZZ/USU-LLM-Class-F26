"""
Send requests to the locally-served fine-tuned model (llama.cpp server)
programmatically, using Python's requests library against its
OpenAI-compatible /v1/chat/completions endpoint.
"""

import requests

SERVER_URL = "http://localhost:8000"


def send_chat_request(
    prompt, server_url=SERVER_URL, max_tokens=150, http_post=requests.post
):
    response = http_post(
        f"{server_url}/v1/chat/completions",
        json={
            "model": "model-q4_k_m",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        },
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


if __name__ == "__main__":  # pragma: no cover
    prompt = "What happens if you crack your knuckles a lot?"
    print(f"Prompt: {prompt}")
    answer = send_chat_request(prompt)
    print(f"Response: {answer}")
