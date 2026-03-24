from pydantic import BaseModel
from typing import List

class Citation(BaseModel):
    # A single source cited in the answer.
    source:str # PDF file name 
    chunk_index:int # which chunk this came from 
    relevance_score : float # Cohere's score for this chunk


class RAGAnswer(BaseModel):
    """
    Structured output from the Generator.
    Holds the answer text plus everything needed to
    explain where the answer came from.
    """
    question: str
    answer: str
    citations: List[Citation]
    model_used: str
    generation_time_seconds: float
    total_chunks_used: int
