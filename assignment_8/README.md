# Assignment 8: vector database

A persistent Chroma vector database of SQuAD paragraphs (`src/vectordb.py`), and an extra-credit RAG step that lets the base Llama answer my ten Assignment 6 questions with and without retrieval (`src/rag.py`).

| File | What it is |
|---|---|
| `src/vectordb.py` | Downloads SQuAD dev-v1.1, keeps the first 200 unique paragraphs, embeds them with Chroma's default MiniLM, stores them on disk with cosine distance, and measures hit@1, hit@3, hit@5 and query latency |
| `tests/test_vectordb.py` | Tests with real Chroma and fake embeddings, so they need no model download and no network |
| `src/rag.py` | Extra credit: puts the ten question notes next to the paragraphs, retrieves the top 3, and asks the base Llama (Ollama `llama3.2:1b`) with one fixed prompt |
| `tests/test_rag.py` | Tests for the prompt, the Ollama request, retrieval, and the saved results |
| `assignment_8/results/rag_results.json` | Raw answers for all 20 runs (10 questions, with and without retrieval) |
| `assignment_8/results/rag_verdicts.json` | My verdicts on those answers, judged with my Assignment 6 correctness guide |
| `assignment_8/screenshots/` | Screenshots of the real run and of ruff and pytest |

Run the vector database (the first run downloads SQuAD and the 79 MB embedding model; later runs reuse the saved database in `chroma_db/`):

    uv run python -m src.vectordb

Run the extra credit (needs Ollama running with `ollama pull llama3.2:1b`):

    uv run python -m src.rag --model llama3.2:1b --out assignment_8/results/rag_results.json

Results, screenshots and my reflection are in the comments on pull request #78.
