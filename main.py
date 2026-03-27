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


def build_retriever():
    """
    Creates and loads the retriever once.
    Call this once at startup and reuse it across multiple queries
    instead of rebuilding the BM25 index and reloading the embedding
    model on every single question.
    """
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
    return retriever


def build_reranker():
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        logger.error("COHERE_API_KEY not found in environment")
        sys.exit(1)

    config = RerankingConfig(
        cohere_api_key=cohere_api_key,
        model=COHERE_MODEL,
        top_n=RERANK_TOP_N,
    )
    return Reranker(config)


def build_generator():
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
    return generator


def run_generation(query: str, retriever=None, reranker=None, generator=None):
    """
    Full pipeline for one query.

    You can pass in pre-built retriever/reranker/generator so they are
    not rebuilt on every call. If you do not pass them, it builds fresh
    ones — useful for single one-off queries.
    """
    if retriever is None:
        retriever = build_retriever()
    if reranker is None:
        reranker = build_reranker()
    if generator is None:
        generator = build_generator()

    logger.info(f"Starting retrieval for: {query}")
    retrieval_artifact  = retriever.retrieve(query)
    retriever.log_results(retrieval_artifact)

    reranking_artifact  = reranker.rerank(retrieval_artifact)
    reranker.log_results(reranking_artifact)

    answer = generator.initiate_generation(reranking_artifact)
    generator.log_results(answer)

    _print_answer(answer)
    return answer


def _print_answer(answer):
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


if __name__ == "__main__":
    RUN_INGESTION = False
    RUN_PIPELINE  = True

    if RUN_INGESTION:
        run_ingestion()

    if RUN_PIPELINE:
        query = "How does the attention mechanism work in transformers?"
        run_generation(query)