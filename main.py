# main.py
# ─────────────────────────────────────────────────────────
# Entry point — creates config, runs pipeline, prints summary
# ─────────────────────────────────────────────────────────

import sys
from rag_docs.config.settings import (
    DOCUMENTS_PATH, CHROMA_DB_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LENGTH,
    EMBEDDING_MODEL, EMBEDDING_DEVICE,
    COLLECTION_NAME, BATCH_SIZE,
)
from rag_docs.entity import IngestionConfig
from rag_docs.core.ingestion import DocumentIngestion
from rag_docs.logging.logger import get_logger
from rag_docs.utils.file_utils import check_folder_exists

logger = get_logger(__name__)


def main():
    print("=" * 50)
    print("   RAG-DOCS — Ingestion Pipeline")
    print("=" * 50)

    # Validate folder
    if not check_folder_exists(DOCUMENTS_PATH, "Documents folder"):
        sys.exit(1)

    # ── Build config ───────────────────────────────────────
    # Fill the config dataclass from settings.py
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

    # ── Create pipeline and run ─────────────────────────────
    pipeline = DocumentIngestion(config)
    artifact = pipeline.initiate_ingestion()

    # ── Print artifact summary ─────────────────────────────
    print("\n" + "=" * 50)
    print("🎉 INGESTION COMPLETE!")
    print(f"   Files processed  : {artifact.total_files_processed}")
    print(f"   Pages loaded     : {artifact.total_pages_loaded}")
    print(f"   Chunks stored    : {artifact.total_chunks_stored}")
    print(f"   ChromaDB saved   : {artifact.chroma_db_path}/")
    print(f"   Collection       : {artifact.collection_name}")
    print(f"   Verification     : {'✅ Passed' if artifact.verification_passed else '❌ Failed'}")
    print(f"   Log file         : {artifact.log_file_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()