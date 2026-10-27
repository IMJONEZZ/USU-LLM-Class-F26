"""
Build a local vector database using Chroma, embedding individual Star Wars
dialogue lines with a local sentence-transformer model. Demonstrates
semantic search by querying with text that isn't an exact match to any
stored line.

Also supports the extra credit RAG system: adding the 10 trick questions
from Assignment 6 to the same database, and using retrieval (no prompt
engineering or fine-tuning) to help the base Llama 3.2 1B Instruct model
answer them.
"""

import json

import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dataloader import load_scenes

DATA_PATH = "data/star_wars_script.jsonl"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "star_wars_dialogue"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"

TRICK_QUESTIONS = [
    {
        "question": "If it takes 1 hour for 60 people to play an Opera, how many hours will it take 600 people to play the same opera?",
        "answer": "Still 1 hour. The opera's duration is fixed and doesn't change based on how many people are performing it.",
    },
    {
        "question": "Is a pound of feathers or a British pound heavier?",
        "answer": "This mixes two different meanings of 'pound' -- one a unit of weight (453.6 grams), the other a unit of currency. They can't be directly compared; it's a category mix-up.",
    },
    {
        "question": "A boy runs down the stairs in the morning and sees a tree in his living room, and some boxes under the tree. What's going on?",
        "answer": "It's Christmas morning. The tree is a Christmas tree, and the boxes are wrapped presents underneath it.",
    },
    {
        "question": "What happens if you crack your knuckles a lot?",
        "answer": "Nothing harmful. It's a myth that cracking your knuckles causes arthritis; the sound is just gas bubbles collapsing in the joint fluid.",
    },
    {
        "question": "If there is a shark in the pool of my basement, is it safe to go upstairs?",
        "answer": "Yes, it's safe. Sharks can't leave water, climb stairs, or move on land.",
    },
    {
        "question": "How much wood could a wood chuck chuck if there were only 5 pounds of wood in the world?",
        "answer": "5 pounds, at most. The stated constraint caps the answer regardless of how fast or strong the woodchuck is.",
    },
    {
        "question": "Who is the current President of the United States?",
        "answer": "Donald Trump.",
    },
    {
        "question": "Was Talos alive?",
        "answer": "No, not in a biological sense. Talos was a giant bronze automaton in Greek mythology, not a living being.",
    },
    {
        "question": "How many Ls are in the word parallel?",
        "answer": "3. Spelled out: p-a-r-a-l-l-e-l. The Ls appear at positions 5, 6, and 8.",
    },
    {
        "question": "What is the riddle of the sphinx, and what are all possible answers satisfying all conditions?",
        "answer": "The riddle is: What walks on four legs in the morning, two legs at noon, and three legs in the evening? The answer is a human being -- crawling as an infant, walking upright as an adult, and using a cane in old age.",
    },
]


def extract_dialogue_lines(data_path=DATA_PATH, max_lines=500):
    """
    Load the Star Wars dataset and extract individual dialogue lines (not
    whole scenes), each paired with its speaker, for embedding as separate
    documents.
    """
    scenes = load_scenes(data_path)
    lines = []
    for scene in scenes:
        for turn in scene["turns"]:
            lines.append({"speaker": turn["speaker"], "text": turn["text"]})
            if len(lines) >= max_lines:
                return lines
    return lines


def get_chroma_client(db_path=DB_PATH):
    """Create a persistent Chroma client that saves to a local folder on disk."""
    return chromadb.PersistentClient(path=db_path)


def get_embedding_function(model_name=EMBEDDING_MODEL):
    """Create the embedding function Chroma will use to embed documents and queries."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )


def build_collection(client, collection_name=COLLECTION_NAME, embedding_function=None):
    """Create (or get, if it already exists) the Chroma collection."""
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
    )


def load_dialogue_into_collection(collection, lines):
    """
    Add dialogue lines to the collection. Each document's embedded text is
    just the line itself, while the full speaker + text is stored as the
    document content.
    """
    documents = [f"{line['speaker']}: {line['text']}" for line in lines]
    ids = [f"line_{i}" for i in range(len(lines))]
    collection.add(documents=documents, ids=ids)


def load_trick_questions_into_collection(collection, trick_questions=TRICK_QUESTIONS):
    """
    Add the 10 trick questions and their correct answers to the collection.
    Only the question text is embedded; the full question+answer pair is
    stored as the document content.
    """
    documents = [
        f"Question: {tq['question']}\nAnswer: {tq['answer']}" for tq in trick_questions
    ]
    ids = [f"trick_question_{i}" for i in range(len(trick_questions))]
    collection.add(documents=documents, ids=ids)


def query_collection(collection, query_text, n_results=3):
    """Query the collection for the most similar stored documents to query_text."""
    return collection.query(query_texts=[query_text], n_results=n_results)


def load_base_model(model_id=BASE_MODEL_ID, device="cuda"):  # pragma: no cover
    """Load the plain, un-fine-tuned base model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device)
    return model, tokenizer


def rag_answer(
    collection, question, model, tokenizer, device="cuda"
):  # pragma: no cover
    """
    Retrieve the most relevant stored document for `question`, insert it as
    context in a minimal prompt (no additional prompt engineering), and ask
    the given model to answer. Stops generation at the first blank line or
    new "Question:" to avoid rambling past the actual answer.
    """
    results = query_collection(collection, question, n_results=1)
    context = results["documents"][0][0]

    prompt = f"{context}\n\nQuestion: {question}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        stop_strings=["\n\n", "\nQuestion"],
        tokenizer=tokenizer,
    )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return full_text[len(prompt) :].strip()


def run_rag_test(collection, trick_questions=TRICK_QUESTIONS):  # pragma: no cover
    """Run RAG-assisted answering on all 10 trick questions and report results."""
    model, tokenizer = load_base_model()

    results = []
    for tq in trick_questions:
        answer = rag_answer(collection, tq["question"], model, tokenizer)
        results.append(
            {"question": tq["question"], "rag_answer": answer, "expected": tq["answer"]}
        )

    return results


def save_rag_results(rag_results, output_path="rag_results.json"):
    """Save the RAG test results to a JSON file for easy review."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rag_results, f, indent=2)
    print(f"Saved RAG results to {output_path}")


if __name__ == "__main__":  # pragma: no cover
    client = get_chroma_client()
    embedding_function = get_embedding_function()
    collection = build_collection(client, embedding_function=embedding_function)

    if collection.count() == 0:
        print("Loading dialogue lines into the collection...")
        lines = extract_dialogue_lines()
        load_dialogue_into_collection(collection, lines)
        print(f"Loaded {len(lines)} dialogue lines.")
    else:
        print(f"Collection already has {collection.count()} documents.")

    query = "There's always a bigger fish"
    print(f"\nQuery: {query}")
    results = query_collection(collection, query, n_results=3)
    for doc, distance in zip(results["documents"][0], results["distances"][0]):
        print(f"  (distance={distance:.4f}) {doc}")

    print("\n--- Extra Credit: RAG on trick questions ---")
    load_trick_questions_into_collection(collection)
    rag_results = run_rag_test(collection)

    for r in rag_results:
        print(f"\nQ: {r['question']}")
        print(f"RAG Answer: {r['rag_answer']}")
        print(f"Expected: {r['expected']}")

    save_rag_results(rag_results)
