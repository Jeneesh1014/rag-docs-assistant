from dataclasses import dataclass
from pathlib import Path


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
    top_k: int = 8