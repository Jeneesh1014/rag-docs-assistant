import os
from dotenv import load_dotenv

from rag_docs.config.settings import (
    CHROMA_DB_PATH,
    DOCUMENTS_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_LENGTH,
    BATCH_SIZE,
)
from rag_docs.entity import IngestionConfig, RetrievalConfig, RerankingConfig
from rag_docs.core.ingestion import DocumentIngestion
from rag_docs.core.retrieval import Retriever
from rag_docs.core.reranking import Reranker
from rag_docs.logging.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


def run_ingestion():
    logger.info("Starting ingestion")
    config = IngestionConfig(
        documents_path=DOCUMENTS_PATH,
        chroma_db_path=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
        embedding_device=EMBEDDING_DEVICE,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        min_chunk_length=MIN_CHUNK_LENGTH,
        batch_size=BATCH_SIZE,
    )
    ingestion = DocumentIngestion(config)
    artifact = ingestion.initiate_ingestion()
    logger.info(
        f"Ingestion complete — {artifact.total_chunks} chunks "
        f"from {artifact.total_documents} documents"
    )
    return artifact


def run_retrieval(query: str):
    logger.info(f"Retrieving chunks for: {query}")
    config = RetrievalConfig(
        chroma_db_path=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
        embedding_device=EMBEDDING_DEVICE,
    )
    retriever = Retriever(config)
    retriever.load_vector_store() 
    retriever.load_bm25_index()       
    artifact = retriever.retrieve(query)
    retriever.log_results(artifact)
    return artifact


def run_reranking(query: str):
    # Retrieve first, then rerank
    retrieval_artifact = run_retrieval(query)

    config = RerankingConfig(
        cohere_api_key=os.getenv("COHERE_API_KEY", ""),
    )

    reranker = Reranker(config)
    artifact = reranker.rerank(retrieval_artifact)
    reranker.log_results(artifact)
    return artifact


if __name__ == "__main__":
    query = "how does the attention mechanism work in transformers"
    run_reranking(query)