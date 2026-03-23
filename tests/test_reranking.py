import os
import time
import pytest
from dotenv import load_dotenv

from rag_docs.config.settings import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
)
from rag_docs.entity import RetrievalConfig, RerankingConfig
from rag_docs.core.retrieval import Retriever
from rag_docs.core.reranking import Reranker

load_dotenv()


#  Shared setup                                                        
@pytest.fixture(scope="module")
def retriever():
    config = RetrievalConfig(
        chroma_db_path=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
        embedding_device=EMBEDDING_DEVICE,
    )
    r = Retriever(config)
    r.load_vector_store()             # ← load ChromaDB + embeddings
    r.load_bm25_index()               # ← build BM25 index
    return r

@pytest.fixture(scope="module")
def reranker():
    config = RerankingConfig(
        cohere_api_key=os.getenv("COHERE_API_KEY", ""),
    )
    r = Reranker(config)
    r.load_client()
    return r



#  Basic sanity tests                                                  
def test_reranker_loads(reranker):
    """Client should initialise without errors."""
    assert reranker.client is not None


def test_output_count(retriever, reranker):
    time.sleep(7)

    """8 chunks in → exactly 3 chunks out."""
    retrieval = retriever.retrieve("what is attention mechanism")
    artifact = reranker.rerank(retrieval)

    assert artifact.total_results == 3
    assert len(artifact.results) == 3


def test_scores_are_valid(retriever, reranker):
    time.sleep(7)

    """Every relevance score should be between 0 and 1."""
    retrieval = retriever.retrieve("how does BERT use transformers")
    artifact = reranker.rerank(retrieval)

    for result in artifact.results:
        assert 0.0 <= result.relevance_score <= 1.0, (
            f"Score out of range: {result.relevance_score} from {result.source}"
        )


def test_results_sorted_by_score(retriever, reranker):
    time.sleep(7)

    """Results must come back best → worst (Cohere guarantees this, we verify it)."""
    retrieval = retriever.retrieve("neural network training optimization")
    artifact = reranker.rerank(retrieval)

    scores = [r.relevance_score for r in artifact.results]
    assert scores == sorted(scores, reverse=True), (
        f"Results not sorted by score: {scores}"
    )


def test_reranked_rank_is_sequential(retriever, reranker):
    time.sleep(7)

    """reranked_rank should be 1, 2, 3 — no gaps."""
    retrieval = retriever.retrieve("what is gradient descent")
    artifact = reranker.rerank(retrieval)

    ranks = [r.reranked_rank for r in artifact.results]
    assert ranks == [1, 2, 3]


def test_original_rank_is_valid(retriever, reranker):
    time.sleep(7)

    """original_rank must be between 1 and top_k (8)."""
    retrieval = retriever.retrieve("convolutional neural networks image classification")
    artifact = reranker.rerank(retrieval)

    for result in artifact.results:
        assert 1 <= result.original_rank <= 8, (
            f"original_rank {result.original_rank} out of expected range"
        )


def test_no_empty_content(retriever, reranker):
    time.sleep(7)

    """No chunk should come back with empty content."""
    retrieval = retriever.retrieve("large language models")
    artifact = reranker.rerank(retrieval)

    for result in artifact.results:
        assert result.content.strip() != "", (
            f"Empty content from {result.source}"
        )


def test_source_is_populated(retriever, reranker):
    time.sleep(7)

    """Every result must have a source filename."""
    retrieval = retriever.retrieve("transformer self attention")
    artifact = reranker.rerank(retrieval)

    for result in artifact.results:
        assert result.source != "", "Source filename is empty"
        assert ".pdf" in result.source.lower(), (
            f"Source doesn't look like a PDF: {result.source}"
        )


def test_reranking_is_fast(retriever, reranker):
    time.sleep(7)

    """Cohere API should respond in under 5 seconds."""
    retrieval = retriever.retrieve("what is reinforcement learning from human feedback")
    start = time.time()
    artifact = reranker.rerank(retrieval)
    elapsed = time.time() - start

    assert elapsed < 5.0, f"Reranking took too long: {elapsed:.2f}s"
    assert artifact.reranking_time_seconds < 5.0


def test_empty_retrieval_handled(reranker):
    """Reranker should not crash if retrieval returns nothing."""
    from rag_docs.entity import RetrievalArtifact

    empty_artifact = RetrievalArtifact(
        query="test",
        results=[],
        total_results=0,
        collection_name="research_papers",
        retrieval_time_seconds=0.0,
    )

    result = reranker.rerank(empty_artifact)
    assert result.total_results == 0
    assert result.results == []



#  Quality tests — does reranking actually improve results?           
def test_transformer_query_sources(retriever, reranker):
    time.sleep(7)

    """
    For a question about attention/transformers, transformer.pdf should
    appear in the top 3 after reranking.
    """
    retrieval = retriever.retrieve("how does multi-head attention work in transformers")
    artifact = reranker.rerank(retrieval)

    sources = [r.source.lower() for r in artifact.results]
    assert any("transformer" in s for s in sources), (
        f"transformer.pdf not in top 3 after reranking. Got: {sources}"
    )


def test_bert_query_sources(retriever, reranker):
    time.sleep(7)

    """
    For a BERT question, bert.pdf should appear in the top 3.
    """
    retrieval = retriever.retrieve("how does BERT use bidirectional training")
    artifact = reranker.rerank(retrieval)

    sources = [r.source.lower() for r in artifact.results]
    assert any("bert" in s for s in sources), (
        f"bert.pdf not in top 3 after reranking. Got: {sources}"
    )


def test_top_result_has_high_score(retriever, reranker):
    time.sleep(7)

    """
    For a clear specific question the top result should score above 0.5.
    If it doesn't, either the docs don't cover this or reranking isn't working.
    """
    retrieval = retriever.retrieve("what is the transformer architecture")
    artifact = reranker.rerank(retrieval)

    top_score = artifact.results[0].relevance_score
    assert top_score > 0.5, (
        f"Top result scored only {top_score} — expected > 0.5 for a clear query"
    )


def test_reranking_changes_order(retriever, reranker):
    time.sleep(7)

    """
    Reranking should change the order of at least some results compared
    to retrieval. If nothing moved, Cohere agreed perfectly with hybrid
    search — possible but worth flagging.
    """
    query = "explain the attention mechanism step by step"
    retrieval = retriever.retrieve(query)
    artifact = reranker.rerank(retrieval)

    # original_rank tells us where each chunk was before reranking
    original_ranks = [r.original_rank for r in artifact.results]

    # If all original_ranks are [1, 2, 3] in order, nothing moved
    order_changed = original_ranks != [1, 2, 3]

    # This is a soft warning, not a hard failure
    # It's valid for Cohere to agree with hybrid search sometimes
    if not order_changed:
        print(
            "\nWARNING: Reranking did not change order for this query. "
            "Cohere agreed with hybrid search exactly."
        )