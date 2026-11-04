"""Generate an HTML answer with Outlines and vLLM on Modal."""

from outlines.types import CFG

import modal

# Force the model to include the HTML tags in the output.
# Only give the model flexibility in title_text, heading_text, and paragraph_text.
HTML_GRAMMAR = r"""
start: doctype html_open head_open meta title_open title_text title_close head_close body_open h1_open heading_text h1_close p_open paragraph_text p_close body_close html_close

doctype: "<!doctype html>"
html_open: "<html lang=\"en\">"
head_open: "<head>"
meta: "<meta charset=\"utf-8\">"
title_open: "<title>"
title_text: HTML_TEXT
title_close: "</title>"
head_close: "</head>"
body_open: "<body>"
h1_open: "<h1>"
heading_text: HTML_TEXT
h1_close: "</h1>"
p_open: "<p>"
paragraph_text: HTML_TEXT
p_close: "</p>"
body_close: "</body>"
html_close: "</html>"

# This regex allows the model to generate any text that does not contain < or >.
# This is to prevent the model from generating additional HTML tags or breaking the structure of the document.
HTML_TEXT: /[^<>]+/
"""

HTML_OUTPUT_FORMAT = CFG(HTML_GRAMMAR)

SYSTEM_PROMPT = """Write a short HTML document with a title, heading, and a single paragraph.
The title should summarize the prompt in two or three words. The heading should
summarize the prompt in a full sentence, stated as a statement rather than a
question. The paragraph should answer the user's question."""

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

# vLLM must be installed before Outlines in this image because of their
# dependency relationship.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm")
    .pip_install("outlines>=1.3.3")
)

app = modal.App("html-answer")


# Create a Modal class to use Modal GPUs for generation.
@app.cls(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("huggingface")],
)
class HtmlAnswerGenerator:
    """Load the model once and serve constrained HTML generations."""

    @modal.enter()
    def load(self) -> None:
        import outlines
        from vllm import LLM

        llm = LLM(
            MODEL_ID,
            dtype="half",
            max_model_len=4096,
            gpu_memory_utilization=0.9,
        )
        self.model = outlines.from_vllm_offline(llm)

    @modal.method()
    def generate(self, question: str) -> str:
        from outlines.inputs import Chat
        from vllm import SamplingParams

        # Combine the system prompt and user question into a single prompt.
        prompt = Chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
        )
        sampling_params = SamplingParams(max_tokens=512)
        # Force the model output to follow the HTML_OUTPUT_FORMAT grammar.
        return self.model(
            prompt,
            HTML_OUTPUT_FORMAT,
            sampling_params=sampling_params,
        )


@app.local_entrypoint()
def main(question: str) -> None:
    """Run with: modal run src/html_generation.py --question 'Your question here'"""
    html = HtmlAnswerGenerator().generate.remote(question)
    print(html)
