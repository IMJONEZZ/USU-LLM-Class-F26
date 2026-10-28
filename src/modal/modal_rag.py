"""Run Chroma retrieval and Llama inference on Modal, then return the answer."""

import modal

LLAMA_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("chromadb", "sentence-transformers", "transformers[torch]")
    .add_local_python_source("src")
)

app = modal.App("llama-rag-vector-db")


def initialize_components():
    """Load the embedding model, collection, tokenizer, and Llama model."""
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.vectordb import MODEL_NAME, create_collection, documents

    embedding_model = SentenceTransformer(MODEL_NAME, device="cpu")
    collection = create_collection(embedding_model, documents)
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_ID,
        torch_dtype=torch.float16,
    ).to("cuda")
    model.eval()
    return embedding_model, collection, tokenizer, model


def build_prompt(context: str, question: str) -> str:
    """Fill the class prompt template with retrieved context and a question."""
    template_text = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful, smart, kind, and efficient AI assistant. "
        "You always fulfill the user's requests to the best of your ability."
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "{context}\n{question}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return template_text.format(context=context, question=question)


def generate_answer(prompt: str, tokenizer, model, device: str = "cuda") -> str:
    """Generate an answer and return only the newly generated text."""
    import torch

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def answer_question(
    question: str, embedding_model, collection, tokenizer, model
) -> str:
    """Retrieve the top document and answer using the assignment prompt."""
    from src.vectordb import query_collection

    matches = query_collection(
        embedding_model,
        collection,
        question,
        n_results=1,
    )
    context = "\n\n".join(str(match["document"]) for match in matches)
    prompt = build_prompt(context, question)
    return generate_answer(prompt, tokenizer, model)


@app.cls(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("huggingface")],
)
class RAG:
    """Load the models and vector database once per Modal container."""

    @modal.enter()
    def load_models_and_documents(self) -> None:
        (
            self.embedding_model,
            self.collection,
            self.tokenizer,
            self.model,
        ) = initialize_components()

    @modal.method()
    def answer(self, question: str) -> str:
        return answer_question(
            question,
            self.embedding_model,
            self.collection,
            self.tokenizer,
            self.model,
        )


@app.local_entrypoint()
def main(question: str) -> None:
    print(RAG().answer.remote(question))
