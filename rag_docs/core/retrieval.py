import time
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from rag_docs.config.settings import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    SEPARATOR_LINE,
)
from rag_docs.entity import RetrievalConfig, RetrievalArtifact, RetrievalResult
from rag_docs.logging.logger import get_logger


class Retriever:

    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.vector_store = None

    def load_vector_store(self) -> None:
        # we are loading an existing db, not creating one
        # so we use Chroma() not Chroma.from_documents()
        self.logger.info("Loading embedding model...")

        embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs={"device": self.config.embedding_device},
        )

        db_path = str(self.config.chroma_db_path)

        if not Path(db_path).exists():
            raise FileNotFoundError(
                f"ChromaDB not found at {db_path} — run ingestion first"
            )

        self.vector_store = Chroma(
            collection_name=self.config.collection_name,
            embedding_function=embeddings,
            persist_directory=db_path,
        )

        count = self.vector_store._collection.count()
        self.logger.info(f"Vector store loaded — {count} chunks available")

    def search(self, query: str) -> list:
        if self.vector_store is None:
            raise RuntimeError("Vector store not loaded — call load_vector_store() first")

        # similarity_search_with_score returns (Document, score) pairs
        # score here is cosine distance — lower means MORE similar
        raw_results = self.vector_store.similarity_search_with_score(
            query=query,
            k=self.config.top_k,
        )

        self.logger.info(f"Raw search returned {len(raw_results)} chunks")
        return raw_results

    def format_results(self, query: str, raw_results: list, elapsed: float) -> RetrievalArtifact:
        results = []

        for i, (doc, score) in enumerate(raw_results):
            # clean up source — some metadata has full path, we want filename only
            source = doc.metadata.get("source", "unknown")
            source = Path(source).name if source != "unknown" else "unknown"

            result = RetrievalResult(
                content=doc.page_content.strip(),
                source=source,
                score=round(float(score), 4),
                chunk_index=i,
            )
            results.append(result)

        return RetrievalArtifact(
            query=query,
            results=results,
            total_results=len(results),
            collection_name=self.config.collection_name,
            retrieval_time_seconds=round(elapsed, 3),
        )

    def retrieve(self, query: str) -> RetrievalArtifact:
        self.logger.info(SEPARATOR_LINE)
        self.logger.info(f"Query: {query}")

        start = time.time()
        raw_results = self.search(query)
        elapsed = time.time() - start

        artifact = self.format_results(query, raw_results, elapsed)

        self.logger.info(f"Retrieved {artifact.total_results} chunks in {elapsed:.3f}s")
        self.logger.info(SEPARATOR_LINE)

        return artifact

    def log_results(self, artifact: RetrievalArtifact) -> None:
        # just for debugging — prints each result clearly
        self.logger.info(f"Results for: '{artifact.query}'")
        self.logger.info(SEPARATOR_LINE)

        for r in artifact.results:
            self.logger.info(f"[{r.chunk_index + 1}] {r.source} — score: {r.score}")
            self.logger.info(f"    {r.content[:150]}...")
            self.logger.info("")