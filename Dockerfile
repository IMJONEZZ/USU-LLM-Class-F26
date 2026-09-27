FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app

# Triton compiles small helper modules at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency metadata first so Docker can cache dependency installation.
COPY pyproject.toml uv.lock .python-version ./

# Recreate the project's locked Python environment inside the image.
RUN uv sync --frozen --no-dev

# Writable runtime locations for non-root execution.
ENV HOME=/tmp
ENV TRITON_CACHE_DIR=/tmp/triton

# Application source will be copied after dependencies.
COPY src/ ./src/

CMD ["/app/.venv/bin/python", "src/gpu_inference.py"]
