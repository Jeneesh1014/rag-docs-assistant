# Ask My Docs — Production RAG Application

A production-grade Retrieval Augmented Generation (RAG) system
for ROS2 and Robotics documentation.

## Status

🚧 Week 1 - In Progress

## Tech Stack

- LangChain + ChromaDB (vector store)
- BM25 + Vector Hybrid Search
- Cohere Reranking
- OpenAI gpt-4o-mini (answer generation)
- Ragas (evaluation)
- GitHub Actions (CI pipeline)

## Architecture

Document Ingestion → Hybrid Search → Reranking → Answer + Citations

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```
