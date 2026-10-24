import os
import re
import time

import httpx
from google import genai
from google.genai import errors, types

from src import config

_TRANSIENT_KINDS = {"rate_limit", "timeout", "server"}
_FENCE = re.compile(r"^```[A-Za-z]*\n(.*?)\n?```$", re.DOTALL)


class TeacherAPIError(RuntimeError):
    def __init__(self, kind, message="", retry_after=None):
        super().__init__(f"{kind}: {message}" if message else kind)
        self.kind = kind
        self.retry_after = retry_after


def get_client():
    key = os.environ.get(config.API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"{config.API_KEY_ENV} is not set; export it before calling Gemini"
        )
    options = types.HttpOptions(timeout=config.TEACHER_TIMEOUT_MS)
    return genai.Client(api_key=key, http_options=options)


def _retry_after(message):
    match = re.search(r"'retryDelay': '(\d+(?:\.\d+)?)s'", message)
    return float(match.group(1)) if match else None


def _client_error_kind(code, message):
    if code in (401, 403):
        return "auth"
    if code == 429:
        return "quota" if "PerDay" in message else "rate_limit"
    return "bad_request"


def _text_or_raise(response):
    feedback = response.prompt_feedback
    if feedback is not None and feedback.block_reason:
        raise TeacherAPIError("blocked", str(feedback.block_reason))
    if not response.candidates:
        raise TeacherAPIError("empty", "no candidates")
    finish = response.candidates[0].finish_reason
    if finish == types.FinishReason.MAX_TOKENS:
        raise TeacherAPIError("truncated")
    if finish not in (None, types.FinishReason.STOP):
        raise TeacherAPIError("blocked", str(finish))
    text = response.text
    if not text or not text.strip():
        raise TeacherAPIError("empty")
    return text


def _call_model(model, prompt, temperature):
    client = get_client()
    settings = types.GenerateContentConfig(
        temperature=temperature, max_output_tokens=config.TEACHER_MAX_OUTPUT_TOKENS
    )
    try:
        response = client.models.generate_content(
            model=model, contents=prompt, config=settings
        )
    except errors.ClientError as e:
        kind = _client_error_kind(e.code, str(e))
        raise TeacherAPIError(kind, str(e), _retry_after(str(e))) from e
    except errors.ServerError as e:
        raise TeacherAPIError("server", str(e)) from e
    except httpx.TimeoutException as e:
        raise TeacherAPIError("timeout", str(e)) from e
    return _text_or_raise(response)


def _wait_too_long(error):
    return (
        error.retry_after is not None
        and error.retry_after > config.TEACHER_MAX_WAIT_SECONDS
    )


def _generate(prompt, model, temperature):
    for _ in range(config.TEACHER_MAX_RETRIES):
        try:
            return _call_model(model, prompt, temperature)
        except TeacherAPIError as e:
            if e.kind not in _TRANSIENT_KINDS or _wait_too_long(e):
                raise
            time.sleep(max(config.TEACHER_RETRY_SECONDS, e.retry_after or 0))
    return _call_model(model, prompt, temperature)


def _unwrap(text):
    text = text.strip()
    fenced = _FENCE.match(text)
    return fenced.group(1).strip() if fenced else text
