import os
import requests
import gradio as gr
from dotenv import load_dotenv
from rag_docs.logging.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

API_URL = os.getenv("API_URL", "http://localhost:8000")
MAX_QUESTION_LENGTH = 500


def format_citations(citations):
    if not citations:
        return "No citations available."

    lines = []
    for i, citation in enumerate(citations, start=1):
        source = citation.get("source", "unknown")
        chunk = citation.get("chunk_index", "?")
        score = citation.get("relevance_score", 0.0)
        lines.append(f"[{i}] {source}  |  chunk {chunk}  |  score {score:.3f}")
    return "\n".join(lines)


def ask(question):
    question = question.strip()

    if not question:
        return "Please enter a question.", ""

    if len(question) > MAX_QUESTION_LENGTH:
        return f"Question too long — please keep it under {MAX_QUESTION_LENGTH} characters.", ""

    logger.info(f"Sending query to API: {question}")

    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question},
            timeout=60,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        logger.error("Could not connect to API")
        return "Could not connect to the API. Make sure api.py is running on port 8000.", ""
    except requests.exceptions.Timeout:
        logger.error("API request timed out")
        return "The request timed out. Please try again.", ""
    except requests.exceptions.HTTPError as e:
        logger.error(f"API returned error: {e}")
        detail = response.json().get("detail", "Unknown error")
        return f"API error: {detail}", ""
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return "Something went wrong. Please try again.", ""

    data = response.json()

    citations_text = format_citations(data.get("citations", []))

    meta = (
        f"\n\n---\n"
        f"Model: {data.get('model_used', 'unknown')}  |  "
        f"Chunks used: {data.get('total_chunks_used', '?')}  |  "
        f"Time: {data.get('generation_time_seconds', 0):.2f}s"
    )

    return data.get("answer", "No answer returned.") + meta, citations_text


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
            with gr.Row():
                submit_btn = gr.Button("Ask", variant="primary")
                clear_btn = gr.Button("Clear")

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

    clear_btn.click(
        fn=lambda: ("", "", ""),
        outputs=[question_box, answer_box, citations_box],
    )


if __name__ == "__main__":
    logger.info("Starting Gradio UI")
    demo.launch()