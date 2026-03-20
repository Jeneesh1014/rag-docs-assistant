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
    SEPARATOR_LINE,
)
from rag_docs.entity import IngestionConfig, RetrievalConfig
from rag_docs.core.ingestion import DocumentIngestion
from rag_docs.core.retrieval import Retriever
from rag_docs.logging.logger import get_logger

logger = get_logger(__name__)


def run_ingestion():
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
    logger.info(f"  documents  : {artifact.total_documents}")
    logger.info(f"  chunks     : {artifact.total_chunks}")
    logger.info(f"  failed     : {artifact.failed_documents}")
    logger.info(f"  collection : {artifact.collection_name}")
    logger.info(SEPARATOR_LINE)


def run_retrieval():
    config = RetrievalConfig(
        chroma_db_path=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
        embedding_device=EMBEDDING_DEVICE,
        top_k=8,
        vector_weight=0.6,
        bm25_weight=0.4,
    )

    retriever = Retriever(config)
    retriever.load_vector_store()
    retriever.load_bm25_index()   # ← new — must call this before hybrid or bm25

    test_queries = [
        "How does the attention mechanism work in transformers?",
        "What is LoRA and how does it reduce training parameters?",
        "How does BERT use masked language modeling?",
    ]

    # run same query through all three methods so you can compare quality
    query = test_queries[0]

    for method in ["vector", "bm25", "hybrid"]:
        artifact = retriever.retrieve(query, method=method)
        retriever.log_results(artifact)


if __name__ == "__main__":
    # comment out run_ingestion() once db exists
    # run_ingestion()
    run_retrieval()