import gradio as gr

import modal

RemoteRAG = modal.Cls.from_name("llama-rag-vector-db", "RAG")
rag = RemoteRAG()


# Add the RAG context to the prompt and send to Modal for answer generation
def answer_question(question: str) -> str:
    return rag.answer.remote(question)


with gr.Blocks(title="RAG Question Answering") as demo:
    gr.Markdown("# RAG Question Answering")

    # Question goes in a textbox, similar to most chatbot interfaces
    question = gr.Textbox(
        label="Your question",
        placeholder="Ask a question...",
        lines=2,
    )
    # Must click the orange "Ask" button to submit the question
    submit = gr.Button("Ask", variant="primary")
    # Answer displays below the textbox in open space
    answer = gr.Markdown(label="Answer")

    # Runs when the user clicks Ask
    submit.click(
        fn=answer_question,
        inputs=question,
        outputs=answer,
    )
    # Runs when the user presses Enter in the textbox
    question.submit(
        fn=answer_question,
        inputs=question,
        outputs=answer,
    )

if __name__ == "__main__":
    demo.launch()
