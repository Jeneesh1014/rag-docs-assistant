import sys
import time
import pytest
from pathlib import Path

# make sure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_docs.config.settings import (
    BM25_WEIGHT,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    TOP_K,
    VECTOR_WEIGHT,
)
from rag_docs.entity import RetrievalConfig, RetrievalArtifact, RetrievalResult
from rag_docs.core.retrieval import Retriever


# ─────────────────────────────────────────────────────────────
# shared retriever — built once, reused across all tests
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def retriever():
    config = RetrievalConfig(
        chroma_db_path=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
        embedding_device=EMBEDDING_DEVICE,
        top_k=TOP_K,
        vector_weight=VECTOR_WEIGHT,
        bm25_weight=BM25_WEIGHT
    )
    r = Retriever(config)
    r.load_vector_store()
    r.load_bm25_index()
    return r


@pytest.fixture(scope="module")
def sample_query():
    return "How does the transformer attention mechanism work?"


# ─────────────────────────────────────────────────────────────
# 1. infrastructure checks
# ─────────────────────────────────────────────────────────────

def test_chroma_db_exists():
    """ingestion must have run before we can test anything"""
    assert CHROMA_DB_PATH.exists(), (
        f"ChromaDB not found at {CHROMA_DB_PATH}. Run ingestion first: python main.py"
    )


def test_chroma_db_not_empty():
    """sanity check — folder should have actual chroma files in it"""
    files = list(CHROMA_DB_PATH.rglob("*"))
    assert len(files) > 0, "ChromaDB folder exists but is empty"


# ─────────────────────────────────────────────────────────────
# 2. retriever loads correctly
# ─────────────────────────────────────────────────────────────

def test_vector_store_loads(retriever):
    """chroma collection should be accessible"""
    assert retriever.vector_store is not None


def test_bm25_index_loads(retriever):
    """bm25 index needs corpus texts and metadata to work"""
    assert retriever.bm25 is not None
    assert len(retriever.bm25_corpus_texts) > 0
    assert len(retriever.bm25_corpus_metadata) > 0


def test_corpus_size_reasonable(retriever):
    """28 PDFs chunked at 500 chars should give us a lot more than 100 chunks"""
    count = len(retriever.bm25_corpus_texts)
    assert count > 100, f"Only {count} chunks found — something went wrong in ingestion"


def test_corpus_metadata_matches_texts(retriever):
    """metadata list and text list must stay in sync — bm25 search depends on this"""
    assert len(retriever.bm25_corpus_texts) == len(retriever.bm25_corpus_metadata)


# ─────────────────────────────────────────────────────────────
# 3. each search method returns results
# ─────────────────────────────────────────────────────────────

def test_vector_search_returns_results(retriever, sample_query):
    artifact = retriever.retrieve(sample_query, method="vector")
    assert artifact.total_results > 0, "Vector search returned nothing"


def test_bm25_search_returns_results(retriever, sample_query):
    artifact = retriever.retrieve(sample_query, method="bm25")
    assert artifact.total_results > 0, "BM25 search returned nothing"


def test_hybrid_search_returns_results(retriever, sample_query):
    artifact = retriever.retrieve(sample_query, method="hybrid")
    assert artifact.total_results > 0, "Hybrid search returned nothing"


# ─────────────────────────────────────────────────────────────
# 4. RetrievalArtifact structure
# ─────────────────────────────────────────────────────────────

def test_artifact_fields_populated(retriever, sample_query):
    artifact = retriever.retrieve(sample_query, method="hybrid")

    assert artifact.query == sample_query
    assert artifact.collection_name == COLLECTION_NAME
    assert artifact.search_method == "hybrid"
    assert artifact.retrieval_time_seconds >= 0
    assert isinstance(artifact.results, list)


def test_result_objects_are_correct_type(retriever, sample_query):
    artifact = retriever.retrieve(sample_query, method="hybrid")
    for result in artifact.results:
        assert isinstance(result, RetrievalResult)


def test_every_result_has_content(retriever, sample_query):
    artifact = retriever.retrieve(sample_query, method="hybrid")
    for i, result in enumerate(artifact.results):
        assert result.content and len(result.content) > 0, (
            f"Result {i} has empty content"
        )


def test_every_result_has_source(retriever, sample_query):
    """source should be a filename — ingestion strips the full path"""
    artifact = retriever.retrieve(sample_query, method="hybrid")
    for i, result in enumerate(artifact.results):
        assert result.source and len(result.source) > 0, (
            f"Result {i} is missing source metadata"
        )


def test_every_result_has_score(retriever, sample_query):
    artifact = retriever.retrieve(sample_query, method="hybrid")
    for i, result in enumerate(artifact.results):
        assert isinstance(result.score, float), (
            f"Result {i} score is not a float: {result.score}"
        )


# ─────────────────────────────────────────────────────────────
# 5. score sanity checks
# ─────────────────────────────────────────────────────────────

def test_hybrid_scores_in_valid_range(retriever, sample_query):
    """hybrid normalizes both scores to [0,1] then combines them
    so final score should stay within [0,1]"""
    artifact = retriever.retrieve(sample_query, method="hybrid")
    for i, result in enumerate(artifact.results):
        assert 0.0 <= result.score <= 1.0, (
            f"Result {i} hybrid score out of range: {result.score}"
        )


def test_top_result_scores_higher_than_last(retriever, sample_query):
    """results should be sorted best-first"""
    artifact = retriever.retrieve(sample_query, method="hybrid")
    if len(artifact.results) < 2:
        pytest.skip("not enough results to compare ordering")

    top = artifact.results[0].score
    last = artifact.results[-1].score
    assert top >= last, (
        f"Results not sorted by score — top: {top:.4f}, last: {last:.4f}"
    )


# ─────────────────────────────────────────────────────────────
# 6. top_k is respected
# ─────────────────────────────────────────────────────────────

def test_vector_search_respects_top_k(retriever, sample_query):
    artifact = retriever.retrieve(sample_query, method="vector")
    assert artifact.total_results <= retriever.config.top_k


def test_bm25_search_respects_top_k(retriever, sample_query):
    artifact = retriever.retrieve(sample_query, method="bm25")
    assert artifact.total_results <= retriever.config.top_k


def test_hybrid_search_respects_top_k(retriever, sample_query):
    artifact = retriever.retrieve(sample_query, method="hybrid")
    assert artifact.total_results <= retriever.config.top_k


# ─────────────────────────────────────────────────────────────
# 7. query relevance — does the right paper come back?
# ─────────────────────────────────────────────────────────────

def test_transformer_query_returns_transformer_paper(retriever):
    """'attention mechanism' should pull transformer.pdf into top results"""
    artifact = retriever.retrieve(
        "What is multi-head self-attention?", method="hybrid"
    )
    sources = [r.source.lower() for r in artifact.results]
    assert any("transformer" in s for s in sources), (
        f"Expected transformer.pdf in results, got: {sources}"
    )


def test_bert_query_returns_bert_paper(retriever):
    artifact = retriever.retrieve(
        "How does BERT use masked language modeling for pretraining?",
        method="hybrid"
    )
    sources = [r.source.lower() for r in artifact.results]
    assert any("bert" in s for s in sources), (
        f"Expected bert.pdf in results, got: {sources}"
    )


def test_rag_query_returns_rag_paper(retriever):
    artifact = retriever.retrieve(
        "How does retrieval augmented generation combine retrieval with generation?",
        method="hybrid"
    )
    sources = [r.source.lower() for r in artifact.results]
    assert any("rag" in s for s in sources), (
        f"Expected rag_paper.pdf in results, got: {sources}"
    )


# ─────────────────────────────────────────────────────────────
# 8. retrieval speed — should not be painfully slow
# ─────────────────────────────────────────────────────────────

def test_hybrid_retrieval_completes_in_reasonable_time(retriever, sample_query):
    """hybrid search on local CPU should finish in under 10 seconds"""
    start = time.time()
    retriever.retrieve(sample_query, method="hybrid")
    elapsed = time.time() - start
    assert elapsed < 10.0, f"Hybrid search took {elapsed:.2f}s — too slow"


# ─────────────────────────────────────────────────────────────
# 9. edge cases
# ─────────────────────────────────────────────────────────────

def test_short_query_does_not_crash(retriever):
    """single word query should not throw an exception"""
    artifact = retriever.retrieve("attention", method="hybrid")
    assert isinstance(artifact, RetrievalArtifact)


def test_long_query_does_not_crash(retriever):
    """very long query should not throw an exception"""
    long_query = (
        "Explain in detail how transformer models use multi-head self-attention "
        "to capture long range dependencies in sequences and how this differs from "
        "recurrent neural networks and LSTM architectures in natural language processing"
    )
    artifact = retriever.retrieve(long_query, method="hybrid")
    assert isinstance(artifact, RetrievalArtifact)


def test_repeated_query_gives_consistent_results(retriever, sample_query):
    """same query twice should return the same top result"""
    artifact1 = retriever.retrieve(sample_query, method="hybrid")
    artifact2 = retriever.retrieve(sample_query, method="hybrid")

    if artifact1.total_results == 0 or artifact2.total_results == 0:
        pytest.skip("no results to compare")

    assert artifact1.results[0].source == artifact2.results[0].source
    assert artifact1.results[0].score == artifact2.results[0].score