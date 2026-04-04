import os
import socket
import sys
from pathlib import Path

import pytest

for _k in list(os.environ):
    if _k.lower() in ("http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(_k, None)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _tcp_ok(host: str, port: int = 443, timeout: float = 3.0) -> bool:
    try:
        socket.create_connection((host, port), timeout=timeout)
        return True
    except OSError:
        return False


NETWORK_AVAILABLE = _tcp_ok("huggingface.co", 443)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live_groq: one real Groq completion (needs network and GROQ_API_KEY)",
    )


def pytest_runtest_setup(item):
    if item.get_closest_marker("live_groq"):
        if not NETWORK_AVAILABLE:
            pytest.skip("Live Groq test needs outbound HTTPS.")
        if not os.getenv("GROQ_API_KEY"):
            pytest.skip("GROQ_API_KEY not set.")


@pytest.fixture(scope="module")
def retriever():
    if not NETWORK_AVAILABLE:
        pytest.skip(
            "Outbound HTTPS unavailable (sentence-transformers loads via Hugging Face). "
            "Run pytest with network for retrieval and reranking tests."
        )
    from rag_docs.config.settings import (
        BM25_WEIGHT,
        CHROMA_DB_PATH,
        COLLECTION_NAME,
        EMBEDDING_MODEL,
        EMBEDDING_DEVICE,
        TOP_K,
        VECTOR_WEIGHT,
    )
    from rag_docs.entity import RetrievalConfig
    from rag_docs.core.retrieval import Retriever

    cfg = RetrievalConfig(
        chroma_db_path=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
        embedding_device=EMBEDDING_DEVICE,
        top_k=TOP_K,
        vector_weight=VECTOR_WEIGHT,
        bm25_weight=BM25_WEIGHT,
    )
    r = Retriever(cfg)
    r.load_vector_store()
    r.load_bm25_index()
    return r
