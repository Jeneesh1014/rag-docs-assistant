<details open>
<summary>Deutsch</summary>

## Überblick

Ask My Docs ist ein produktionsnahes Retrieval-Augmented Generation (RAG) System zur Beantwortung von Fragen aus AI/ML-Forschungspapieren.

Das System kombiniert Retrieval, Reranking und Generation, um Antworten zu liefern, die auf echten Dokumenten basieren. Dadurch werden Halluzinationen reduziert und die Zuverlässigkeit verbessert.

---

## Aktueller Stand

- Woche 1: Ingestion und Hybrid Retrieval — 26 Tests bestanden
- Woche 2: Reranking und vollständige Pipeline — 51 Tests bestanden
- Woche 3: Evaluationspipeline mit Ragas + GitHub Actions CI Quality Gate
- End-to-End Validierung: 10/10 Anfragen korrekt mit Zitaten beantwortet
- CI-Status: Erfolgreich mit gespeicherten Evaluationsergebnissen

---

## Hauptfunktionen

- Hybrid-Suche (BM25 + Vektor) für bessere Genauigkeit
- Reranking mit Cohere zur Verbesserung der Ergebnisqualität
- Antworten mit Zitaten aus den Quelldokumenten
- Strukturierte Ausgaben mit Pydantic
- Evaluationspipeline mit automatischen Qualitätsprüfungen (Ragas + CI)
- Wiederverwendbare Pipeline-Komponenten (Retriever, Reranker, Generator)

---

## Evaluationsergebnisse

Datensatz: 20 Fragen basierend auf AI/ML-Forschungspapieren

- Faithfulness: 0.734 (über Schwellenwert)
- Context Precision: 0.808
- Context Recall: 0.350
- Answer Relevancy: übersprungen (Groq Einschränkung)

Das System besteht das definierte Quality Gate und wird über GitHub Actions validiert.

---

## Tech Stack

- LangChain und ChromaDB für Retrieval
- sentence-transformers (all-MiniLM-L6-v2) für lokale Embeddings
- BM25 + Vektor-Suche für Hybrid Retrieval
- Cohere rerank-english-v3.0 für Reranking
- Groq LLaMA Modelle für Generation
- Ragas für Evaluation
- FastAPI und Gradio (geplant) für API und UI

---

## Implementierungsdetails

Die Pipeline folgt dem Ablauf: retrieve → rerank → generate.

Hybrid Retrieval kombiniert Keyword- und semantische Suche.  
Reranking verbessert die Qualität der ausgewählten Kontexte vor der Generierung.

Zitate werden direkt aus den Retrieval-Ergebnissen erzeugt und nicht vom LLM extrahiert, was die Zuverlässigkeit erhöht.

Die Evaluation ist in den Workflow integriert und wird über ein CI Quality Gate überwacht.

---

## Performance

- Retrieval: ~0.09 Sekunden
- Reranking: ~0.37 Sekunden
- Generation: ~0.69 Sekunden

Das System liefert schnelle und gleichzeitig nachvollziehbare Antworten.

</details>

---

<details>
<summary>English</summary>

## Overview

Ask My Docs is a production-ready Retrieval-Augmented Generation (RAG) system designed to answer questions from AI/ML research papers.

The system combines retrieval, reranking, and generation to produce answers grounded in source documents, reducing hallucinations and improving reliability.

---

## Current Progress

- Week 1: Ingestion and Hybrid Retrieval — 26 tests passing
- Week 2: Reranking and Full Pipeline — 51 tests passing
- Week 3: Evaluation pipeline with Ragas + GitHub Actions CI quality gate
- End-to-end validation: 10/10 queries verified with correct citations
- CI status: Passing with committed evaluation results

---

## Key Features

- Hybrid search (BM25 + vector) for better recall and precision
- Reranking using Cohere to improve final context quality
- Answers grounded with inline citations from source documents
- Structured outputs using Pydantic models
- Evaluation pipeline with automated quality checks (Ragas + CI)
- Reusable pipeline components (retriever, reranker, generator)

---

## Evaluation Results

Dataset: 20 questions based on AI/ML research papers

- Faithfulness: 0.734 (passes threshold)
- Context Precision: 0.808
- Context Recall: 0.350
- Answer Relevancy: skipped (Groq limitation)

The system passes the defined quality gate and is validated through GitHub Actions.

---

## Tech Stack

- LangChain and ChromaDB for retrieval pipeline
- sentence-transformers (all-MiniLM-L6-v2) for local embeddings
- BM25 + vector search for hybrid retrieval
- Cohere rerank-english-v3.0 for reranking
- Groq LLaMA models for generation
- Ragas for evaluation
- FastAPI and Gradio (planned) for API and UI

---

## Notes on Implementation

The pipeline follows a structured flow: retrieve → rerank → generate.

Hybrid retrieval balances keyword matching and semantic understanding.  
Reranking improves the quality of selected context before generation.

Citations are created directly from retrieved results instead of parsing LLM output, making them more reliable.

Evaluation is integrated into the workflow and enforced through a CI quality gate.

---

## Performance

- Retrieval: ~0.09 seconds
- Reranking: ~0.37 seconds
- Generation: ~0.69 seconds

The system provides fast and reliable responses with grounded outputs.

</details>
