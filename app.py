import gradio as gr
from dotenv import load_dotenv

from main import build_retriever, build_reranker, build_generator, run_generation
from rag_docs.logging.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

# Build once at startup — not rebuilt per query
retriever = build_retriever()
reranker = build_reranker()
generator = build_generator()

logger.info("Pipeline ready")


def format_citations(answer):
    if not answer.citations:
        return "No citations available."

    lines = []
    for i, citation in enumerate(answer.citations, start=1):
        lines.append(
            f"[{i}] {citation.source}  |  chunk {citation.chunk_index}  |  score {citation.relevance_score:.3f}"
        )
    return "\n".join(lines)


def ask(question):
    question = question.strip()
    if not question:
        return "Please enter a question.", ""

    logger.info(f"Query received: {question}")

    try:
        answer = run_generation(question, retriever, reranker, generator)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return f"Something went wrong: {e}", ""

    citations_text = format_citations(answer)

    meta = (
        f"\n\n---\n"
        f"Model: {answer.model_used}  |  "
        f"Chunks used: {answer.total_chunks_used}  |  "
        f"Time: {answer.generation_time_seconds:.2f}s"
    )

    return answer.answer + meta, citations_text


EXAMPLES = [
    "How does the attention mechanism work in transformers?",
    "What is the difference between BERT and GPT?",
    "How does dropout help prevent overfitting?",
    "What are the key ideas behind reinforcement learning from human feedback?",
]

with gr.Blocks(title="Ask My Docs") as demo:
    gr.Markdown("# Ask My Docs\nAsk questions about AI & ML research papers.")

    with gr.Row():
        with gr.Column(scale=2):
            question_box = gr.Textbox(
                label="Your question",
                placeholder="e.g. How does the attention mechanism work in transformers?",
                lines=2,
            )
            submit_btn = gr.Button("Ask", variant="primary")

    with gr.Row():
        with gr.Column(scale=2):
            answer_box = gr.Markdown(label="Answer")

    with gr.Row():
        with gr.Column(scale=2):
            citations_box = gr.Textbox(
                label="Citations",
                lines=4,
                interactive=False,
            )

    gr.Examples(
        examples=EXAMPLES,
        inputs=question_box,
        label="Example questions",
    )

    submit_btn.click(
        fn=ask,
        inputs=question_box,
        outputs=[answer_box, citations_box],
    )

    question_box.submit(
        fn=ask,
        inputs=question_box,
        outputs=[answer_box, citations_box],
    )


if __name__ == "__main__":
    logger.info("Starting Gradio UI")
    demo.launch()