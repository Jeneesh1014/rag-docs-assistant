import os

import gradio as gr
import requests
from dotenv import load_dotenv

from rag_docs.logging.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
MAX_QUESTION_LENGTH = 500


def _format_citations(citations):
    if not citations:
        return "No citations returned."

    lines = []
    for i, citation in enumerate(citations, start=1):
        source = citation.get("source", "unknown")
        chunk = citation.get("chunk_index", "?")
        score = citation.get("relevance_score", 0.0)
        lines.append(f"[{i}] {source}  |  chunk {chunk}  |  score {score:.3f}")
    return "\n".join(lines)


def ask(question):
    question = (question or "").strip()

    if not question:
        return "Enter a question first.", ""

    if len(question) > MAX_QUESTION_LENGTH:
        return (
            f"Question is too long (max {MAX_QUESTION_LENGTH} characters).",
            "",
        )

    logger.info("POST /ask via UI")

    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question},
            timeout=90,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        logger.error("Cannot reach API at %s", API_URL)
        return (
            "Cannot reach the API. Start the stack with `python app/run.py` "
            "and ensure port 8000 is free.",
            "",
        )
    except requests.exceptions.Timeout:
        logger.error("Request timed out")
        return "The request timed out. Try a shorter question.", ""
    except requests.exceptions.HTTPError:
        logger.error("HTTP error from API: %s", response.status_code)
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        return f"API error: {detail}", ""
    except Exception as e:
        logger.exception("Unexpected UI error")
        return f"Something went wrong: {e}", ""

    data = response.json()
    citations_text = _format_citations(data.get("citations", []))

    meta = (
        "\n\n---\n"
        f"Model: {data.get('model_used', 'unknown')}  |  "
        f"Chunks: {data.get('total_chunks_used', '?')}  |  "
        f"Time: {float(data.get('generation_time_seconds', 0)):.2f}s"
    )

    body = data.get("answer", "") or "No answer in response."
    return body + meta, citations_text


EXAMPLES = [
    "How does the attention mechanism work in transformers?",
    "What is the difference between BERT and GPT?",
    "How does dropout help prevent overfitting?",
    "What are the main ideas behind reinforcement learning from human feedback?",
    "What is multi-head attention and why is it used?",
]

with gr.Blocks(title="Ask My Docs") as demo:
    gr.Markdown(
        "# Ask My Docs\n"
        "Questions are answered from the ingested PDF corpus via the local API."
    )

    with gr.Row():
        with gr.Column(scale=2):
            question_box = gr.Textbox(
                label="Question",
                placeholder="e.g. How does scaled dot-product attention work?",
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
                lines=5,
                interactive=False,
            )

    gr.Examples(
        examples=EXAMPLES,
        inputs=question_box,
        label="Example questions",
    )

    submit_btn.click(fn=ask, inputs=question_box, outputs=[answer_box, citations_box])
    question_box.submit(fn=ask, inputs=question_box, outputs=[answer_box, citations_box])
    clear_btn.click(
        fn=lambda: ("", "", ""),
        outputs=[question_box, answer_box, citations_box],
    )
