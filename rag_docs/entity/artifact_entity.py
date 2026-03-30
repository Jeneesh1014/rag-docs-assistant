from dataclasses import dataclass, field
from typing import List


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
    search_method:str = "hybrid"


@dataclass
class RerankingResult:
    content: str
    source: str
    relevance_score: float   
    original_rank: int       
    reranked_rank: int       

@dataclass
class RerankingArtifact:
    query: str
    results: List[RerankingResult]
    total_results: int
    model: str
    reranking_time_seconds: float


@dataclass
class EvaluationSample:
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str


@dataclass
class EvaluationArtifact:
    total_questions: int
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    results_path: str
    samples: List[EvaluationSample] = field(default_factory=list)

    def passes_quality_gate(self, min_faithfulness: float = 0.7) -> bool:
        return self.faithfulness >= min_faithfulness