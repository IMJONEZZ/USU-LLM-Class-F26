"""FastAPI service: POST /generate returns an HTML page and whether it is valid.

Run for real (Apple GPU, model loads once at startup):
    uv run --group model uvicorn src.app:create_default_app --factory
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.html_service import MAX_NEW_TOKENS, Generator, generate_page


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    constrained: bool = True
    max_new_tokens: int = Field(default=MAX_NEW_TOKENS, ge=50, le=1000)


class GenerateResponse(BaseModel):
    prompt: str
    constrained: bool
    html: str
    valid: bool
    errors: list[str]
    seconds: float


def create_app(model: Generator) -> FastAPI:
    """Build the service around any ``Generator`` (real model or a test fake)."""
    app = FastAPI(title="HTML structured output service")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest) -> dict:
        return generate_page(
            model, request.prompt, request.constrained, request.max_new_tokens
        )

    return app


def create_default_app() -> FastAPI:  # pragma: no cover
    """Uvicorn factory: loads the real Qwen model once, then serves."""
    from src.html_service import load_model

    return create_app(load_model())
