from dataclasses import dataclass
from pathlib import Path
from rag_docs.config.settings import (
    TOP_K, VECTOR_WEIGHT, BM25_WEIGHT,
    COHERE_MODEL, RERANK_TOP_N,
    GROQ_MODEL, MAX_TOKENS, TEMPERATURE,
)


@dataclass
class IngestionConfig:
    documents_path: Path
    chroma_db_path: Path
    chunk_size: int
    chunk_overlap: int
    min_chunk_length: int
    embedding_model: str
    embedding_device: str
    collection_name: str
    batch_size: int


@dataclass
class RetrievalConfig:
    chroma_db_path: Path
    collection_name: str
    embedding_model: str
    embedding_device: str
    top_k: int = TOP_K
    vector_weight: float = VECTOR_WEIGHT
    bm25_weight: float = BM25_WEIGHT


@dataclass
class RerankingConfig:
    cohere_api_key: str
    model: str = COHERE_MODEL
    top_n: int = RERANK_TOP_N


@dataclass
class GenerationConfig:
    groq_api_key: str
    model: str = GROQ_MODEL
    temperature: float = TEMPERATURE
    max_tokens: int = MAX_TOKENS