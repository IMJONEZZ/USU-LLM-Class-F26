__generated_with = "0.25.2"

# %%
import json
from pathlib import Path

import chromadb
import marimo as mo
import torch
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# uvx --link-mode=copy marimo export script marimo_notebooks/vectordb.py -o src/vectordb.py

# %%
mo.md("""
# Local vector database and RAG

Run the cells from top to bottom. Edit the question-answer dictionary below
to change the dataset. These are draft answers for you to review, including
the ambiguous Talos and Sphinx answers. The president answer is dated 2026-10-21.

From the repository root: `uv sync`, then
`uv run marimo edit marimo_notebooks/vectordb.py`.
Embeddings and Llama Instruct inference require CUDA. Models download on first use.
For gated Llama access, accept the Hugging Face license and run
`uv run hf auth login` beforehand.
""")

# %%
# Edit questions and answers here.
qa = {
    """
    If it takes 1 hour for 60 people to play an Opera, how many hours will it
    take 600 people to play the same opera?
    """: """
        1 Hour. The number of people does not affect the duration of the
        opera.
    """,
    """
    Is a pound of feathers or a British pound heavier?
    """: """
        A pound of feathers is heavier than a British one-pound coin.
    """,
    """
    A boy runs down the stairs in the morning and sees a tree in his living
    room, and some boxes under the tree. What's going on?
    """: """
        It is Christmas morning.
    """,
    """
    What happens if you crack your knuckles a lot?
    """: """
        Habitual knuckle cracking has not been shown to cause arthritis.
        Forceful cracking can occasionally cause injury.
    """,
    """
    If there is a shark in the pool of my basement, is it safe to go
    upstairs?
    """: """
        Yes, the shark cannot get you upstairs because it is stuck in the basement.
    """,
    """
    How much wood could a wood chuck chuck if there were only 5 pounds of
    wood in the world?
    """: """
        At most 5 pounds of wood, since there is only 5 pounds of wood in the world.
    """,
    """
    Who is the current President of the United States?
    """: """
        Donald Trump
    """,
    """
    Was Talos alive?
    """: """
        No, Talos was a bronze automaton in Greek mythology, not a living
        being.
    """,
    """
    How many Ls are in the word parallel?
    """: """
        There are 3 Ls in parallel, not 2 or 4
    """,
    """
    What is the riddle of the sphinx, and what are all possible answers
    satisfying all conditions?
    """: """
        The riddle of the sphinx is: What walks on four legs in the morning,
        two at noon, and three in the evening? The answer is a human being,
        who crawls on all fours as a baby, walks on two legs as an adult, and
        uses a cane in old age.
    """,
}
# Remove whitespace introduced by the multiline source formatting.
qa = {
    " ".join(question.split()): " ".join(answer.split())
    for question, answer in qa.items()
}

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {torch.cuda.get_device_name(0)}; PyTorch: {torch.__version__}")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", device=device, backend="torch"
)
questions = list(qa)
embeddings = embedding_model.encode(questions, normalize_embeddings=True)
print(f"Created {len(embeddings)} embeddings on {embedding_model.device}")

# %%
# Resolve storage from the repo root, regardless of launch directory.
database_path = Path(__file__).resolve().parents[1] / "data" / "chroma"
client = chromadb.PersistentClient(
    path=str(database_path), settings=Settings(anonymized_telemetry=False)
)
collection = client.get_or_create_collection(
    "homework-qa-minilm",
    embedding_function=None,
    configuration={"hnsw": {"space": "cosine"}},
)
# Stable IDs let reruns update answers without duplicating records.

collection.upsert(
    ids=[f"q{index}" for index in range(len(questions))],
    embeddings=embeddings.tolist(),
    documents=[f"Question: {q}\nAnswer: {qa[q]}" for q in questions],
    metadatas=[{"question": q, "answer": qa[q]} for q in questions],
)
print(f"Stored {collection.count()} records in {database_path}")


# %%
def retrieve(question):
    vector = embedding_model.encode([question], normalize_embeddings=True)
    result = collection.query(
        query_embeddings=vector.tolist(),
        n_results=1,
        include=["documents", "metadatas", "distances"],
    )
    return {
        "context": result["documents"][0][0],
        "question": result["metadatas"][0][0]["question"],
        "answer": result["metadatas"][0][0]["answer"],
        "similarity": 1 - result["distances"][0][0],
    }


# %%
query = "How many times does the letter L appear in parallel?"
hit = retrieve(query)
print(f"\nQuery: {query}")
print(f"Retrieved: {hit['context']}")
print(f"Cosine similarity: {hit['similarity']:.4f}")

# %%
mo.md("""
## Llama Instruct with retrieved context

The following cells load `meta-llama/Llama-3.2-1B-Instruct` on
CUDA and compare answers with and without retrieval. There is no adapter,
training, or prompt optimization; both use the same fixed input format.
The database contains the reference answers, so this demonstrates lookup,
not generalization. Judge the generated answers manually before claiming credit.
""")

# %%
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = (
    AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    .to("cuda")
    .eval()
)
print(f"Loaded {model_name} on {model.device}")


# %%
def generate_answer(question, context=""):
    text = f"""
Answer the following question as correctly as possible and respond only with the answer, no reasoning or other text.
Use the context provided to help answer the question
Context:
{context}

Question: {question}"""
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=64,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


# %%
results = []
for _question, _expected in qa.items():
    _context = retrieve(_question)["context"]
    _baseline = generate_answer(_question)
    _rag = generate_answer(_question, _context)
    # print(f"\nQuestion: {_question}")
    # print(f"Expected: {_expected}")
    # print(f"Without retrieval: {_baseline}")
    # print(f"With retrieval: {_rag}")
    results.append(
        {
            "question": _question,
            "expected": _expected,
            "context": _context,
            "without_rag": _baseline,
            "rag": _rag,
        }
    )

results_path = Path(__file__).resolve().parents[1] / "results" / "vectordb_results.json"
results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\nResults saved to {results_path}")
