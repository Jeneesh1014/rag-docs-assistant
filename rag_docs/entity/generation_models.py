from typing import List

from pydantic import BaseModel


class Citation(BaseModel):
    source: str
    chunk_index: int
    relevance_score: float


class RAGAnswer(BaseModel):
    question: str
    answer: str
    citations: List[Citation]
    model_used: str
    generation_time_seconds: float
    total_chunks_used: int
