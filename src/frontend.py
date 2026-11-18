import gradio as gr

import modal

RemoteRAG = modal.Cls.from_name("llama-rag-vector-db", "RAG")
rag = RemoteRAG()


def answer_question(question: str) -> str:
    return rag.answer.remote(question)


with gr.Blocks(title="RAG Question Answering") as demo:
    gr.Markdown("# RAG Question Answering")

    question = gr.Textbox(
        label="Your question",
        placeholder="Ask a question...",
        lines=2,
    )
    submit = gr.Button("Ask", variant="primary")
    answer = gr.Markdown(label="Answer")

    submit.click(
        fn=answer_question,
        inputs=question,
        outputs=answer,
    )
    question.submit(
        fn=answer_question,
        inputs=question,
        outputs=answer,
    )

if __name__ == "__main__":
    demo.launch()
