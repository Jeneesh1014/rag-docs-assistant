# Ask My Docs — Research Assistant for AI/ML Papers (Production RAG System)

End-to-end retrieval stack over a local PDF corpus: hybrid BM25 and vector search, Cohere reranking, Groq generation with Pydantic-grounded citations, Ragas metrics, CI gate, FastAPI, and Gradio.

**Live demo:** https://huggingface.co/spaces/JeneeshSurani/rag-docs-assistant

Try asking:

- "Explain attention mechanism in transformers"
- "What is contrastive learning?"

## Architecture

Pipeline flow:

1. PDF ingestion → chunking → embeddings → stored in ChromaDB
2. Query → hybrid retrieval (BM25 + vector search)
3. Top results → Cohere reranker
4. Final context → Groq LLM (Llama 3.1)
5. Output → structured answer with citations

The system ensures:

- Retrieval is prioritized over generation
- Answers are grounded in real document chunks
- Citations are always traceable to source text

## What It Does

This project is a research assistant for AI/ML papers.

Users can ask questions in natural language, and the system:

- Retrieves relevant sections from research papers
- Ranks the most useful information
- Generates a grounded answer with citations

The system is designed to reduce hallucinations by forcing the model
to answer only from retrieved context.

Example:
"How does attention work in transformers?"
→ Returns a structured answer with sources from actual papers.

## Why This Project

Large Language Models often hallucinate and generate incorrect information.

This project focuses on:

- Improving factual accuracy using retrieval (RAG)
- Measuring answer quality with Ragas
- Building a reliable, testable AI system

It demonstrates how to move from "demo AI" to "production-ready AI".

## Technical highlights

- Hybrid search with weighted fusion (BM25 0.4, dense 0.6) in `rag_docs/core/retrieval.py`
- Reranking with `rerank-english-v3.0` before generation
- Pydantic models for answers and citations (`rag_docs/entity/generation_models.py`)
- Ragas evaluation and committed `evaluation_results.json`; GitHub Actions runs `check_results.py` as a quality gate
- FastAPI (`POST /ask`, `GET /health`, OpenAPI at `/docs`) plus Gradio UI that calls the API over HTTP
- Retriever, reranker, and generator built once at API startup (FastAPI lifespan)

## Key Design Decisions

- Hybrid Search (BM25 + Dense):
  Improves recall compared to using only vector search

- Reranking before generation:
  Ensures only the most relevant chunks are passed to the LLM

- Pydantic for structured outputs:
  Prevents malformed responses and ensures citation integrity

- CI Quality Gate:
  Prevents performance regression using evaluation metrics

- FastAPI + Gradio separation:
  Keeps backend and UI loosely coupled for scalability

## Evaluation results

Committed run (`evaluation_results.json`, 20 questions). Answer relevancy is skipped for this setup (Groq batch constraint).

| Metric            | Score |
| ----------------- | ----- |
| Faithfulness      | 0.73  |
| Answer relevancy  | n/a   |
| Context precision | 0.81  |
| Context recall    | 0.35  |

Quality gate: faithfulness minimum 0.7 (passing in the committed file).

## Example outputs

Excerpted from the `samples` field in `evaluation_results.json` (wording may differ slightly on a live run after retrieval noise).

**Q:** "How does the attention mechanism work in transformers?"

**A:** "According to the Transformer paper [1], the attention mechanism allows modeling dependencies without regard to distance in the input or output sequences [2, 19]…" (answer continues with encoder/decoder self-attention and Figure 3.)

**Sources:** chunk text in the eval trace maps to filenames in `data/documents/` (e.g. _Attention Is All You Need_). **Time:** generation alone is typically sub-second on Groq; end-to-end latency includes retrieval and reranking.

## Tech stack

| Component  | Technology                    |
| ---------- | ----------------------------- |
| LLM        | Llama 3.1 8B Instant via Groq |
| Embeddings | all-MiniLM-L6-v2 (local)      |
| Vector DB  | ChromaDB (local persistence)  |
| Search     | BM25 plus dense hybrid        |
| Reranking  | Cohere rerank-english-v3.0    |
| Evaluation | Ragas                         |
| API        | FastAPI, Uvicorn              |
| UI         | Gradio                        |
| CI         | GitHub Actions                |

## Project structure

```
rag-docs-assistant/
├── app/
│   ├── api.py           # FastAPI app and routes
│   ├── ui.py            # Gradio UI (HTTP client to API)
│   └── run.py           # API on :8000 and UI on :7860
├── data/
│   ├── documents/       # PDF corpus
│   └── eval_questions.json
├── rag_docs/
│   ├── config/settings.py
│   ├── entity/
│   ├── logging/logger.py
│   ├── utils/file_utils.py
│   └── core/            # ingestion, retrieval, reranking, generation, evaluation
├── tests/
│   └── conftest.py      # shared retriever fixture; skips if no outbound HTTPS
├── .github/workflows/eval.yml
├── main.py              # CLI pipeline entry
├── run_evaluation.py
├── check_results.py     # CI quality gate reader
├── evaluation_results.json
├── start.sh
├── requirements.txt
├── .env.example
└── README.md
```

## How to run

```bash
git clone <your-fork-url>
cd RAG-DOC-ASSISTANT
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # add GROQ_API_KEY and COHERE_API_KEY
```

Ingest and persist the vector store (run when PDFs or chunk settings change):

```bash
python main.py --ingest
```

One-off question through the pipeline (loads models each time unless you wire reuse yourself):

```bash
python main.py
```

API plus UI (from repository root):

```bash
python -m app.run
# same behavior:
python app/run.py
```

- API and Swagger: http://127.0.0.1:8000/docs
- Gradio: http://127.0.0.1:7860

API only (no Gradio): `uvicorn app.api:app --host 0.0.0.0 --port 8000`

Shell helper:

```bash
chmod +x start.sh
./start.sh
```

## Tests and evaluation

`tests/conftest.py` clears `HTTP(S)_PROXY` (broken proxies often break Hugging Face hub) and probes `huggingface.co:443`. If that fails, retrieval and reranking tests are **skipped** instead of erroring; generation unit tests, empty-rerank handling, and `tests/test_app.py` still run.

`test_real_groq_call` is marked `live_groq`: it runs only with outbound HTTPS and a set `GROQ_API_KEY`; otherwise it is skipped.

```bash
pytest tests/               # 52 tests with network; subset passes when offline
python run_evaluation.py    # needs keys; writes evaluation_results.json
```

Manual end-to-end smoke (rate-limited on free Cohere tier):

```bash
python tests/test_manual_pipeline.py
```

## License and data

Use and license of bundled PDFs follow their original publications; this repo is a technical demo only.
