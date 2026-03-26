# main.py

import os
import sys
from dotenv import load_dotenv

from rag_docs.config.settings import (
    DOCUMENTS_PATH,
    CHROMA_DB_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_LENGTH,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    COLLECTION_NAME,
    BATCH_SIZE,
    TOP_K,
    VECTOR_WEIGHT,
    BM25_WEIGHT,
    RERANK_TOP_N,
    COHERE_MODEL,
    GROQ_MODEL,
    TEMPERATURE,
    MAX_TOKENS,
    SEPARATOR_LINE,
)
from rag_docs.entity import (
    IngestionConfig,
    RetrievalConfig,
    RerankingConfig,
    GenerationConfig,
)
from rag_docs.core.ingestion import DocumentIngestion
from rag_docs.core.retrieval import Retriever
from rag_docs.core.reranking import Reranker
from rag_docs.core.generation import Generator
from rag_docs.logging.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


# ─────────────────────────────────────────────
# INGESTION
# ─────────────────────────────────────────────

def run_ingestion():
    logger.info("Starting ingestion")

    config = IngestionConfig(
        documents_path=DOCUMENTS_PATH,
        chroma_db_path=CHROMA_DB_PATH,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        min_chunk_length=MIN_CHUNK_LENGTH,
        embedding_model=EMBEDDING_MODEL,
        embedding_device=EMBEDDING_DEVICE,
        collection_name=COLLECTION_NAME,
        batch_size=BATCH_SIZE,
    )

    ingestion = DocumentIngestion(config)
    artifact = ingestion.initiate_ingestion()

    logger.info(SEPARATOR_LINE)
    logger.info("Ingestion complete")
    logger.info(f"Documents processed: {artifact.total_documents}")
    logger.info(f"Total chunks stored: {artifact.total_chunks}")
    logger.info(f"Failed documents: {artifact.failed_documents}")
    logger.info(f"Collection: {artifact.collection_name}")
    logger.info(SEPARATOR_LINE)

    return artifact


# ─────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────

def run_retrieval(query: str):
    logger.info(f"Starting retrieval for: {query}")

    config = RetrievalConfig(
        chroma_db_path=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
        embedding_device=EMBEDDING_DEVICE,
        top_k=TOP_K,
        vector_weight=VECTOR_WEIGHT,
        bm25_weight=BM25_WEIGHT,
    )

    retriever = Retriever(config)
    retriever.load_vector_store()
    retriever.load_bm25_index()

    artifact = retriever.retrieve(query)
    retriever.log_results(artifact)

    return artifact


# ─────────────────────────────────────────────
# RERANKING
# ─────────────────────────────────────────────

def run_reranking(query: str):
    retrieval_artifact = run_retrieval(query)

    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        logger.error("COHERE_API_KEY not found in environment")
        sys.exit(1)

    config = RerankingConfig(
        cohere_api_key=cohere_api_key,
        model=COHERE_MODEL,
        top_n=RERANK_TOP_N,
    )

    reranker = Reranker(config)
    artifact = reranker.rerank(retrieval_artifact)
    reranker.log_results(artifact)

    return artifact


# ─────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────

def run_generation(query: str):
    """
    Full pipeline: retrieve → rerank → generate.
    Returns a RAGAnswer with citations.
    """
    reranking_artifact = run_reranking(query)

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.error("GROQ_API_KEY not found in environment")
        sys.exit(1)

    config = GenerationConfig(
        groq_api_key=groq_api_key,
        model=GROQ_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    generator = Generator(config)
    generator.load_client()

    answer = generator.initiate_generation(reranking_artifact)
    generator.log_results(answer)

    # Pretty print to terminal so Day 6 manual testing is easy to read
    _print_answer(answer)

    return answer


def _print_answer(answer):
    """
    Human-readable terminal output for manual testing.
    Not a logger call — this is intentional UI output for the developer.
    """
    print()
    print(SEPARATOR_LINE)
    print(f"Question : {answer.question}")
    print(SEPARATOR_LINE)
    print(f"Answer   :\n{answer.answer}")
    print()
    print("Citations:")
    for c in answer.citations:
        print(f"  [{c.chunk_index}] {c.source} — relevance: {c.relevance_score:.3f}")
    print()
    print(
        f"Model: {answer.model_used} | "
        f"Time: {answer.generation_time_seconds:.1f}s | "
        f"Chunks used: {answer.total_chunks_used}"
    )
    print(SEPARATOR_LINE)
    print()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── Change this flag to control what runs ──
    RUN_INGESTION = False   # Set True only when you need to re-ingest
    RUN_PIPELINE  = True    # Full retrieve → rerank → generate

    if RUN_INGESTION:
        run_ingestion()

    if RUN_PIPELINE:
        # Day 6 question bank — uncomment one at a time or loop all
        query = "How does the attention mechanism work in transformers?"
        run_generation(query)