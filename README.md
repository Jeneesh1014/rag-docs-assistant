# Ask My Docs — Production RAG Application

A production-grade RAG system for querying AI & Machine Learning research papers.
Ask questions in plain English — get cited answers backed by real papers.

## Status

✅ Week 1 Complete — Ingestion + Hybrid Retrieval + 26 Tests Passing

## Tech Stack

- **LangChain + ChromaDB** — vector store (local, no API needed)
- **sentence-transformers/all-MiniLM-L6-v2** — embeddings (local CPU)
- **BM25 + Vector Hybrid Search** — keyword + semantic, weighted 0.6/0.4
- **Cohere rerank-english-v3.0** — reranking top results
- **Groq llama-3.3-70b-versatile** — answer generation
- **Ragas** — retrieval + generation evaluation
- **GitHub Actions** — CI quality gates
- **Gradio + FastAPI** — UI and API endpoint

## Pipeline
