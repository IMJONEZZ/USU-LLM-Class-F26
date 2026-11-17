"""Serves the Gradio front end as a Modal ASGI app.

Same reason the optimizer ran on Modal in assignment 10. The API key lives in
a secret, and running Gradio on my laptop would mean exporting it into a
shell, which is the road that ended with me rotating a key in assignment 7.
Here the secret is injected into the container and the browser talks to a URL.

No GPU on this one. It holds a web server and makes HTTP calls to the two
services that do have GPUs, so it's the cheapest container Modal offers and it
scales to zero like everything else.

    modal deploy gpu/frontend_app.py
"""

import modal

PORT_CONCURRENCY = 8
SCALEDOWN_SECONDS = 300

# Modal cancels an input at 300 seconds by default, and a request that
# arrives while the GPU behind it is still loading can legitimately take
# longer than that. Without this the container kills its own request and
# the browser just waits forever.
REQUEST_TIMEOUT = 900

# Pinned to what I developed against. Gradio 6 dropped the `type` parameter
# that every ChatInterface tutorial still passes, so a floating version here
# would be a working deploy today and a TypeError whenever it drifts.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("gradio==6.32.0", "dspy==3.4.1")
    .add_local_dir("src", "/root/src")
)

app = modal.App("hw11-frontend", image=image)


@app.function(
    secrets=[modal.Secret.from_name("hw7-api-key")],
    scaledown_window=SCALEDOWN_SECONDS,
    timeout=REQUEST_TIMEOUT,
    max_containers=1,
)
@modal.concurrent(max_inputs=PORT_CONCURRENCY)
@modal.asgi_app()
def ui():
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    from src.frontend import build_app

    return mount_gradio_app(app=FastAPI(), blocks=build_app(), path="/")
