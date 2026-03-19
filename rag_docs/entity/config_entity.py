from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class IngestionArtifact:
    total_documents: int
    total_chunks: int
    failed_documents: int
    collection_name: str
    chroma_db_path: str
    embedding_model: str


@dataclass
class RetrievalResult:
    # one single chunk that came back from search
    content: str
    source: str
    score: float
    chunk_index: int


@dataclass
class RetrievalArtifact:
    query: str
    results: List[RetrievalResult]
    total_results: int
    collection_name: str
    retrieval_time_seconds: float