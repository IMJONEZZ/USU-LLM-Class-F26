"""Retrieval in front of the base model, for the extra credit.

The rules are no fine tuning and no prompt engineering, and the model has to be
the stock Llama 3.2 1B, not the one I trained in assignment 6. So the
only moving part that's allowed to help is retrieval: look the question up in
the vector database, put what comes back in front of the model, and ask.

The template below is four lines and I'm leaving it that way. Tuning the
wording until the answers improve is exactly the thing the assignment rules
out, and it would also make the result meaningless, because then I'd be
measuring my prompt instead of the retrieval.

Everything is injected. The client, the encoder and the HTTP session all come
in as arguments, which is what lets the tests run this end to end without a
database, a model download, or a server.

    uv run python -m src.rag
"""

from src.inference import ask
from src.vectordb import QA_COLLECTION, search

BASE_URL = "https://froostysnoowman--hw8-serve-base-serve.us-east.modal.run"
BASE_MODEL = "llama-3.2-1b-base"

# One document. There are only ten in the collection and a question matches its
# own entry almost exactly, so pulling three would mean handing the model two
# answers to questions nobody asked. On a corpus this small that's noise, and a
# 1B has enough trouble with the right context without adding wrong context.
TOP_K = 1

PROMPT = """Context:
{context}

Question: {question}
Answer:"""


def format_context(hits):
    """Render retrieved documents as plain question and answer text."""
    blocks = []
    for hit in hits:
        question, answer = hit.get("question"), hit.get("answer")
        if question and answer:
            blocks.append(f"Q: {question}\nA: {answer}")
        elif answer:
            blocks.append(answer)
        else:
            blocks.append(hit.get("text", ""))
    return "\n\n".join(b for b in blocks if b)


def build_prompt(question, hits):
    return PROMPT.format(context=format_context(hits), question=question)


def retrieve(question, client, encoder, name=QA_COLLECTION, limit=TOP_K):
    return search(client, question, encoder, name=name, limit=limit)


def answer_question(
    question,
    client,
    encoder,
    url=BASE_URL,
    limit=TOP_K,
    session=None,
    key=None,
    **kwargs,
):
    """Retrieve, then ask the base model, and report both halves.

    The retrieved document comes back alongside the answer because the two
    failure modes look identical from the outside. If the answer is wrong I
    need to know whether retrieval handed it the wrong document or whether it
    had the right one and ignored it.
    """
    hits = retrieve(question, client, encoder, limit=limit)
    prompt = build_prompt(question, hits)
    reply = ask(
        prompt,
        url=url,
        session=session,
        key=key,
        model=BASE_MODEL,
        **kwargs,
    )
    return {
        "question": question,
        "answer": reply,
        "retrieved": [h.get("question", h.get("text", "")) for h in hits],
        "retrieval_correct": bool(hits) and hits[0].get("question") == question,
    }


if __name__ == "__main__":  # pragma: no cover - entry point, not logic
    import json

    from src.dataset import HOMEWORK_QUESTIONS
    from src.vectordb import build_qa, connect, load_encoder

    client, encoder = connect(), load_encoder()
    print(f"indexed {build_qa(client, encoder)} question/answer pairs\n")

    results = []
    for i, question in enumerate(HOMEWORK_QUESTIONS, 1):
        result = answer_question(question, client, encoder, max_tokens=400)
        results.append(result)
        flag = "" if result["retrieval_correct"] else "   <- retrieved the wrong one"
        print(f"Q{i}. {question}")
        print(f"    retrieved: {result['retrieved'][0][:70]}{flag}")
        print(f"    answer:    {result['answer']}\n")

    hit_rate = sum(r["retrieval_correct"] for r in results)
    print(f"retrieval found the right document for {hit_rate} of {len(results)}")

    with open("data/rag.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    client.close()
