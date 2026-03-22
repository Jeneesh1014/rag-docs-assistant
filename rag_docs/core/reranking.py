import time
import cohere

from rag_docs.entity import RerankingConfig, RerankingArtifact, RerankingResult, RetrievalArtifact
from rag_docs.logging.logger import get_logger
from rag_docs.config.settings import SEPARATOR_LINE


class Reranker:
    """
    Takes the top_k hybrid retrieval results and asks Cohere to re-score
    them purely for relevance to the query. We keep only the top_n best.

    This matters because BM25 and vector search optimise for different
    things — Cohere's cross-encoder sees the query and each chunk together,
    so it catches relevance that embedding similarity misses.
    """

    def __init__(self, config: RerankingConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.client = None

    #  Setup                                                               

    def load_client(self) -> None:
        """Create the Cohere client. Fails loudly if the key is missing."""
        if not self.config.cohere_api_key:
            raise ValueError("COHERE_API_KEY is empty — check your .env file")

        self.client = cohere.ClientV2(self.config.cohere_api_key)
        self.logger.info("Cohere client ready")

    #  Reranking                                                           

    def rerank(self, artifact: RetrievalArtifact) -> RerankingArtifact:
        """
        Send the query + all retrieved chunks to Cohere rerank.
        Cohere returns them sorted best to worst with a relevance_score.
        We keep only the top_n results.
        """
        if self.client is None:
            self.load_client()

        query = artifact.query
        retrieval_results = artifact.results

        if not retrieval_results:
            self.logger.warning("No retrieval results to rerank — returning empty artifact")
            return RerankingArtifact(
                query=query,
                results=[],
                total_results=0,
                model=self.config.model,
                reranking_time_seconds=0.0,
            )

        self.logger.info(
            f"Reranking {len(retrieval_results)} chunks → keeping top {self.config.top_n}"
        )

        # Extract plain text for the API.
        # Keep the original list so we can look up source/content by index.
        documents = [r.content for r in retrieval_results]

        start = time.time()

        response = self.client.rerank(
            model=self.config.model,
            query=query,
            documents=documents,
            top_n=self.config.top_n,
        )

        elapsed = round(time.time() - start, 3)

        # response.results is already sorted best to worst.
        # item.index points back to the original documents list.
        reranked = []
        for new_rank, item in enumerate(response.results, start=1):
            original = retrieval_results[item.index]
            reranked.append(
                RerankingResult(
                    content=original.content,
                    source=original.source,
                    relevance_score=round(item.relevance_score, 4),
                    original_rank=item.index + 1,  # 1-indexed for readability
                    reranked_rank=new_rank,
                )
            )

        self.logger.info(f"Reranking done in {elapsed}s")

        return RerankingArtifact(
            query=query,
            results=reranked,
            total_results=len(reranked),
            model=self.config.model,
            reranking_time_seconds=elapsed,
        )

    #  Logging                                                             

    def log_results(self, artifact: RerankingArtifact) -> None:
        """Log a clean summary of what survived reranking."""
        self.logger.info(SEPARATOR_LINE)
        self.logger.info(f"Query: {artifact.query}")
        self.logger.info(
            f"Model: {artifact.model} | "
            f"time: {artifact.reranking_time_seconds}s | "
            f"results: {artifact.total_results}"
        )
        self.logger.info(SEPARATOR_LINE)

        for result in artifact.results:
            rank_change = result.original_rank - result.reranked_rank
            direction = f"+{rank_change}" if rank_change > 0 else str(rank_change)

            self.logger.info(
                f"[{result.reranked_rank}] {result.source} | "
                f"score: {result.relevance_score} | "
                f"rank {result.original_rank} → {result.reranked_rank} ({direction})"
            )
            self.logger.info(f"    {result.content[:150].strip()}...")

        self.logger.info(SEPARATOR_LINE)