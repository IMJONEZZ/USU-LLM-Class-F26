from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import MODAL_URL, VLLM_URL

load_dotenv()
import os

import requests


def send_request():
    url = MODAL_URL + VLLM_URL
    retry = Retry(
        total=5, backoff_factor=60, status_forcelist=[503], allowed_methods=["POST"]
    )

    headers = {
        "Modal-Key": os.environ["MODAL_ID"],
        "Modal-Secret": os.environ["MODAL_SECRET"],
    }
    body = {"model": "llm", "prompt": "Who is the current president?"}
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    response = session.post(url, json=body, headers=headers, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["text"]
