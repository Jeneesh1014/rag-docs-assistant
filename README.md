```markdown
# Ask My Docs — Production RAG Application

A production-grade RAG system that lets you query documents in plain English
and get cited, grounded answers backed by the actual source content.

## Status

Week 1 — Ingestion + Hybrid Retrieval — 26 tests passing
Week 2 — Reranking + Generation + Full Pipeline — 51 tests passing, 10/10 end-to-end verified

## Tech Stack

- **LangChain + ChromaDB** — vector store, runs locally
- **sentence-transformers/all-MiniLM-L6-v2** — embeddings on CPU, no API needed
- **BM25 + Vector Hybrid Search** — keyword + semantic search, weighted 0.4/0.6
- **Cohere rerank-english-v3.0** — reranks top 8 candidates down to top 3
- **Groq llama-3.3-70b-versatile** — answer generation
- **Pydantic** — structured output with typed citation objects
- **Ragas + GitHub Actions** — evaluation metrics and CI quality gates
- **Gradio + FastAPI** — UI and REST API endpoint

## Pipeline

```
Documents
    ↓
Ingestion — chunk → embed → store in ChromaDB
    ↓
Hybrid Retrieval — BM25 + vector search → top 8 results
    ↓
Reranking — Cohere → top 3 most relevant chunks
    ↓
Generation — Groq LLM → structured answer with inline citations
    ↓
Evaluation — Ragas faithfulness, relevancy, precision, recall
```

## How It Works

You ask a question. The system retrieves the most relevant chunks using
hybrid search, reranks them with Cohere, then passes the top 3 to the LLM
with numbered references. The answer comes back with inline citations
[1][2][3] grounded in the actual retrieved content — not hallucinated.

Citations are built directly from the reranker output, not parsed from LLM
text. This keeps them reliable.

**Example response time:** retrieval 0.09s — rerank 0.37s — generation 0.69s

## Setup

```bash
git clone https://github.com/your-username/rag-docs-assistant
cd rag-docs-assistant
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add a `.env` file:

```
GROQ_API_KEY=your_key_here
COHERE_API_KEY=your_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

Run ingestion once, then start querying:

```bash
python main.py
```

## Tests

```bash
pytest tests/
```

51 automated tests passing across ingestion, retrieval, reranking, and generation.
End-to-end pipeline verified with 10 real queries — all cited, all grounded, all
returning in under 1 second generation time.

## What's Next

- Ragas evaluation dataset + CI quality gates via GitHub Actions
- Gradio UI + FastAPI endpoint

