# Ask My Docs — Production RAG Application

A production-ready Retrieval Augmented Generation (RAG) system that answers questions about AI & ML research papers. Built with a hybrid search pipeline, reranking, structured citation generation, automated evaluation, and a web UI with REST API.

---

## Architecture

```
User Question
      ↓
  Gradio UI (port 7860)
      ↓  HTTP POST /ask
  FastAPI Backend (port 8000)
      ↓
┌─────────────────────────────────┐
│         RAG Pipeline            │
│                                 │
│  1. Hybrid Retrieval            │
│     BM25 (0.4) + Vector (0.6)   │
│     → top 8 chunks              │
│                                 │
│  2. Reranking                   │
│     Cohere rerank-english-v3.0  │
│     → top 3 chunks              │
│                                 │
│  3. Generation                  │
│     Groq llama-3.1-8b-instant   │
│     → structured cited answer   │
└─────────────────────────────────┘
      ↓
  Answer + Citations + Metadata
```

---

## Evaluation Results

Evaluated on 20 questions across 28 AI/ML research papers using Ragas.

| Metric            | Score | Threshold |
|-------------------|-------|-----------|
| Faithfulness      | 0.734 | ≥ 0.70 ✅ |
| Context Precision | 0.808 | —         |
| Context Recall    | 0.350 | —         |

> Answer relevancy is skipped — Groq only supports `n=1` and Ragas requires `n>1` for this metric.

Quality gate: **PASS** — enforced automatically via GitHub Actions CI on every push.

---

## Tech Stack

| Component     | Technology                              |
|---------------|-----------------------------------------|
| LLM           | Groq — llama-3.1-8b-instant             |
| Embeddings    | sentence-transformers/all-MiniLM-L6-v2 (local, CPU) |
| Vector DB     | ChromaDB (local)                        |
| Keyword Search| rank-bm25                               |
| Reranking     | Cohere rerank-english-v3.0              |
| Evaluation    | Ragas                                   |
| Backend API   | FastAPI + Uvicorn                       |
| Frontend UI   | Gradio                                  |
| CI            | GitHub Actions                          |

All components are free. No OpenAI.

---

## Project Structure

```
rag-docs-assistant/
├── data/
│   ├── documents/                  ← 28 PDF research papers
│   └── eval_questions.json         ← 20 evaluation questions + ground truths
├── rag_docs/
│   ├── config/settings.py          ← all constants
│   ├── entity/                     ← dataclasses + Pydantic models
│   ├── logging/logger.py           ← structured logging
│   └── core/
│       ├── ingestion.py            ← PDF loading, chunking, embedding, ChromaDB
│       ├── retrieval.py            ← hybrid BM25 + vector search
│       ├── reranking.py            ← Cohere reranker
│       ├── generation.py           ← Groq LLM + structured output
│       └── evaluation.py          ← Ragas evaluation pipeline
├── tests/                          ← 51 automated tests
├── .github/workflows/eval.yml      ← CI quality gate
├── main.py                         ← full pipeline entry point
├── run_evaluation.py               ← runs Ragas evaluation
├── check_results.py                ← reads results, exits 0 or 1 for CI
├── api.py                          ← FastAPI REST endpoint
├── app.py                          ← Gradio web UI
├── start.sh                        ← launches both servers together
└── evaluation_results.json         ← committed results read by CI
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-docs-assistant.git
cd rag-docs-assistant
```

### 2. Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```
GROQ_API_KEY=your_groq_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
```

Get your keys here — both are free:
- Groq: https://console.groq.com
- Cohere: https://dashboard.cohere.com

### 5. Add your PDF documents

Place your PDF files in `data/documents/`.

### 6. Run ingestion

This only needs to be done once. It loads all PDFs, chunks them, embeds them, and stores them in ChromaDB.

```bash
python main.py --ingest
```

---

## How to Run

### Option A — Both servers with one command

```bash
./start.sh
```

- Gradio UI → http://localhost:7860
- FastAPI → http://localhost:8000
- API docs → http://localhost:8000/docs

Press `Ctrl+C` to stop both.

### Option B — Run separately

**Terminal 1 — API:**
```bash
source venv/bin/activate
uvicorn api:app --port 8000
```

**Terminal 2 — UI:**
```bash
source venv/bin/activate
python app.py
```

### Run the full pipeline from terminal

```bash
python main.py
```

### Run evaluation

```bash
python run_evaluation.py
```

### Check evaluation results

```bash
python check_results.py
```

---

## API Usage

### Health check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "pipeline": {
    "retriever": true,
    "reranker": true,
    "generator": true
  }
}
```

### Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does the attention mechanism work in transformers?"}'
```

```json
{
  "question": "How does the attention mechanism work in transformers?",
  "answer": "According to the Transformer paper, the attention mechanism...",
  "citations": [
    {
      "source": "transformer_paper.pdf",
      "chunk_index": 1,
      "relevance_score": 0.979
    },
    {
      "source": "transformer_paper.pdf",
      "chunk_index": 2,
      "relevance_score": 0.896
    },
    {
      "source": "transformer_paper.pdf",
      "chunk_index": 3,
      "relevance_score": 0.562
    }
  ],
  "model_used": "llama-3.1-8b-instant",
  "generation_time_seconds": 0.83,
  "total_chunks_used": 3
}
```

Interactive API docs available at `http://localhost:8000/docs`.

---

## Tests

```bash
pytest tests/
```

```
tests/test_pipeline.py        26 passed
tests/test_reranking.py       14 passed
tests/test_generation.py      11 passed

Total: 51 passed
```

---

## CI — GitHub Actions

Every push to `main` triggers the quality gate:

1. Checks out the repository
2. Reads `evaluation_results.json` (committed to repo)
3. Fails the build if faithfulness drops below 0.70
4. Prints scores summary

No API calls are made in CI — the pipeline runs locally and results are committed.

---

## Sample Output

```
Question : How does the attention mechanism work in transformers?

Answer   : According to the Transformer paper, the attention mechanism allows
           modeling of dependencies without regard to their distance in the
           input or output sequences...

Citations:
  [1] transformer_paper.pdf  |  chunk 1  |  score 0.979
  [2] transformer_paper.pdf  |  chunk 2  |  score 0.896
  [3] transformer_paper.pdf  |  chunk 3  |  score 0.562

Model    : llama-3.1-8b-instant
Time     : retrieval 0.091s  |  rerank 0.371s  |  generation 0.69s
```

---

## License

MIT
