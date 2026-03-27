# tests/test_manual_pipeline.py

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from main import run_generation, build_retriever, build_reranker, build_generator


QUESTIONS = [
    "How does the attention mechanism work in transformers?",
    "What is multi-head attention and why is it better than single-head attention?",
    "What makes BERT different from GPT in terms of pre-training objectives?",
    "How does the Vision Transformer apply transformers to image classification?",
    "What techniques are used to prevent overfitting in deep learning?",
    "How does batch normalization improve neural network training?",
    "How does a diffusion model generate images from noise?",
    "What is the vanishing gradient problem and how is it addressed in RNNs?",
    "What is retrieval-augmented generation and why is it useful?",
    "Explain how large language models are fine-tuned on specific tasks.",
]


def run_all():
    print("\n" + "=" * 60)
    print("  Pipeline Manual Test — 10 Questions")
    print("=" * 60)

    # Build once, reuse across all questions
    # This means the embedding model loads once and BM25 index builds once
    print("\n  Loading models (once)...\n")
    retriever = build_retriever()
    reranker  = build_reranker()
    generator = build_generator()

    summary = []

    for i, question in enumerate(QUESTIONS, start=1):
        print(f"\n[{i}/{len(QUESTIONS)}] {question}\n")

        try:
            answer = run_generation(
                question,
                retriever=retriever,
                reranker=reranker,
                generator=generator,
            )

            answer_ok = len(answer.answer.strip()) > 80
            has_cites = len(answer.citations) > 0
            time_ok   = answer.generation_time_seconds < 8.0
            has_refs  = any(
                f"[{c.chunk_index}]" in answer.answer
                for c in answer.citations
            )

            status = "PASS" if (answer_ok and has_cites and time_ok) else "REVIEW"

            summary.append({
                "num":      i,
                "status":   status,
                "cites":    len(answer.citations),
                "refs":     "yes" if has_refs else "no",
                "time":     f"{answer.generation_time_seconds:.1f}s",
                "question": question[:50],
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            summary.append({
                "num":      i,
                "status":   "FAIL",
                "cites":    0,
                "refs":     "no",
                "time":     "—",
                "question": question[:50],
            })

        # Cohere free tier — 10 calls/minute
        if i < len(QUESTIONS):
            print("\n  (waiting 8s...)\n")
            time.sleep(8)

    print("\n" + "=" * 60)
    print(f"  {'#':<4} {'Status':<8} {'Cites':<7} {'Refs':<6} {'Time':<8} Question")
    print(f"  {'-'*4} {'-'*8} {'-'*7} {'-'*6} {'-'*8} {'-'*25}")
    for r in summary:
        print(
            f"  {r['num']:<4} {r['status']:<8} {r['cites']:<7} "
            f"{r['refs']:<6} {r['time']:<8} {r['question']}"
        )

    passed = sum(1 for r in summary if r["status"] == "PASS")
    print(f"\n  {passed}/{len(QUESTIONS)} passed")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all()