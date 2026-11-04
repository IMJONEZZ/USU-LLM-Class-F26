"""FastAPI service for Assignment 9 constrained HTML generation."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.html_generator import (
    HtmlGenerationError,
    HtmlGenerator,
    build_default_generator,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_PROMPT_CHARS = 4000


class GenerateHtmlRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_CHARS)
    # Length is controlled by Guidance gen(max_tokens=...) / repeat(max=...)
    # in html_grammar.py — not a DIY max_new_tokens knob.
    temperature: float = Field(0.7, ge=0.0, le=2.0)

    model_config = {"extra": "forbid"}


def create_app(
    generator_loader: Callable[[], HtmlGenerator] | None = None,
) -> FastAPI:
    # Tests pass a fake loader so CI never downloads Qwen.
    loader = generator_loader or build_default_generator

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load (or mock) once at startup; tear down on shutdown.
        app.state.generator = loader()
        yield
        app.state.generator = None

    app = FastAPI(title="Assignment 9 — Constrained HTML", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> dict[str, str]:
        generator = getattr(request.app.state, "generator", None)
        return {
            "status": "ready" if generator is not None else "starting",
            "service": "generate-html",
        }

    @app.post("/generate-html")
    def generate_html(payload: GenerateHtmlRequest, request: Request) -> Response:
        generator: HtmlGenerator = request.app.state.generator
        generator.temperature = payload.temperature
        try:
            html = generator.generate(payload.prompt)
        except HtmlGenerationError as exc:
            # Bad prompt / incomplete HTML / grammar mismatch → client error.
            logger.warning("generate-html failed: %s", exc)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("generate-html crashed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        # Raw HTML on purpose — assignment asks for structured HTML output,
        # not JSON wrapping.
        return Response(content=html, media_type="text/html; charset=utf-8")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        index_path = STATIC_DIR / "index.html"
        if not index_path.is_file():
            return HTMLResponse(
                "<p>Assignment 9 HTML demo. POST /generate-html</p>",
                status_code=200,
            )
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app


# Uvicorn entry: ``uvicorn src.server:app``
app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("src.server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":  # pragma: no cover
    main()
