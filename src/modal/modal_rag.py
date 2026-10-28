"""Run Chroma retrieval and Llama inference on Modal, then return the answer."""  # pragma: no cover

import modal

EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
LLAMA_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("chromadb", "sentence-transformers", "transformers[torch]")
    .add_local_python_source("src")
)

app = modal.App("llama-rag-vector-db")


@app.cls(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("huggingface")],
)
class RAG:
    """Load the models and vector database once per Modal container."""

    @modal.enter()
    def load_models_and_documents(self) -> None:
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from src.vectordb import MODEL_NAME, create_collection, documents

        self.embedding_model = SentenceTransformer(MODEL_NAME, device="cpu")
        self.collection = create_collection(self.embedding_model, documents)

        self.tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL_ID,
            torch_dtype=torch.float16,
        ).to("cuda")
        self.model.eval()

    @modal.method()
    def answer(self, question: str) -> str:
        import torch

        from src.vectordb import query_collection

        matches = query_collection(
            self.embedding_model,
            self.collection,
            question,
            n_results=1,
        )
        context = "\n\n".join(str(match["document"]) for match in matches)

        template_text = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful, smart, kind, and efficient AI assistant. "
            "You always fulfill the user's requests to the best of your ability."
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "{context}\n{question}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        prompt = template_text.format(context=context, question=question)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).to("cuda")

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@app.local_entrypoint()
def main(question: str) -> None:
    print(RAG().answer.remote(question))
