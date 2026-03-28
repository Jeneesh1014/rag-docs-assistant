## Overview

Ask My Docs is a production-ready Retrieval-Augmented Generation (RAG) system designed to provide accurate answers from document collections.

The system retrieves relevant content and ensures that every response is supported by actual source data, reducing hallucinations and improving reliability.

---

## Current Progress

- Week 1: Ingestion and Hybrid Retrieval — 26 tests passing  
- Week 2: Reranking and Full Pipeline — 51 tests passing  
- End-to-end validation: 10/10 queries verified with correct citations  

---

## Key Features

- Hybrid search combining semantic and keyword-based retrieval  
- Inline citations grounded in retrieved documents  
- Fast response time with sub-second generation  
- Structured outputs using typed schemas  
- Fully tested pipeline with evaluation support  

---

## Tech Stack

- LangChain and ChromaDB for vector storage and retrieval  
- sentence-transformers (all-MiniLM-L6-v2) for embeddings  
- BM25 and vector search for hybrid retrieval  
- Cohere rerank model for improved relevance  
- Groq LLaMA models for answer generation  
- Pydantic for structured outputs  

---

## Notes on Implementation

The system uses a hybrid retrieval approach to balance keyword matching and semantic understanding. Retrieved results are reranked before being passed to the language model.

Citations are generated directly from the retrieval pipeline rather than relying on the model output, ensuring consistency and correctness.

---

## Performance

- Retrieval: ~0.09 seconds  
- Reranking: ~0.37 seconds  
- Generation: ~0.69 seconds  

