import time
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from rag_docs.config.settings import SEPARATOR_LINE
from rag_docs.entity import RetrievalConfig, RetrievalArtifact, RetrievalResult
from rag_docs.logging.logger import get_logger


class Retriever:

    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.vector_store = None

        # these get populated in load_bm25_index()
        self.bm25_index = None
        self.bm25_corpus_texts = []    # raw text for each stored chunk
        self.bm25_corpus_metadata = [] # matching metadata for each chunk


    # SETUP — load vector store + build BM25 index

    def load_vector_store(self) -> None:
        # loading an existing db — Chroma(), not Chroma.from_documents()
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

    def load_bm25_index(self) -> None:
        # BM25 needs all chunk texts upfront — pull them from ChromaDB
        # we do this once at load time, not on every query
        if self.vector_store is None:
            raise RuntimeError("Load vector store first before building BM25 index")

        self.logger.info("Building BM25 index from stored chunks...")

        # _collection.get() returns all docs — texts + metadata
        all_docs = self.vector_store._collection.get(
            include=["documents", "metadatas"]
        )

        self.bm25_corpus_texts = all_docs["documents"]
        self.bm25_corpus_metadata = all_docs["metadatas"]

        # tokenize — BM25Okapi just needs a list of token lists
        # simple whitespace split is fine here, BM25 is keyword matching anyway
        tokenized = [text.lower().split() for text in self.bm25_corpus_texts]
        self.bm25_index = BM25Okapi(tokenized)

        self.logger.info(f"BM25 index built — {len(self.bm25_corpus_texts)} chunks indexed")


    # SEARCH — three modes: vector, bm25, hybrid

    def vector_search(self, query: str) -> list:
        # returns list of (text, metadata, score) tuples
        # score is cosine distance — lower means MORE similar
        if self.vector_store is None:
            raise RuntimeError("Vector store not loaded")

        raw = self.vector_store.similarity_search_with_score(
            query=query,
            k=self.config.top_k,
        )

        # convert to plain tuples so format_results can handle both search types
        return [(doc.page_content, doc.metadata, score) for doc, score in raw]

    def bm25_search(self, query: str) -> list:
        # returns list of (text, metadata, score) tuples
        # score is BM25 score — higher means MORE relevant
        if self.bm25_index is None:
            raise RuntimeError("BM25 index not built — call load_bm25_index() first")

        tokens = query.lower().split()
        scores = self.bm25_index.get_scores(tokens)

        # pair each score with its chunk index, sort best first, take top_k
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top = ranked[:self.config.top_k]

        results = []
        for idx, score in top:
            text = self.bm25_corpus_texts[idx]
            metadata = self.bm25_corpus_metadata[idx]
            results.append((text, metadata, score))

        return results

    def hybrid_search(self, query: str) -> list:
        # combines vector and BM25 scores into a single ranked list
        # returns list of (text, metadata, combined_score) tuples

        vector_results = self.vector_search(query)
        bm25_results = self.bm25_search(query)

        # ---- normalize vector scores ----
        # chroma returns cosine distance (0=identical, 2=opposite)
        # flip it so higher = better, then clamp to [0, 1]
        vector_sims = [max(0.0, 1.0 - score) for _, _, score in vector_results]

        v_max = max(vector_sims) if vector_sims else 1.0
        v_min = min(vector_sims) if vector_sims else 0.0
        v_range = v_max - v_min if v_max != v_min else 1.0
        vector_norm = [(s - v_min) / v_range for s in vector_sims]

        # ---- normalize BM25 scores ----
        bm25_scores = [score for _, _, score in bm25_results]
        b_max = max(bm25_scores) if bm25_scores else 1.0
        b_min = min(bm25_scores) if bm25_scores else 0.0
        b_range = b_max - b_min if b_max != b_min else 1.0
        bm25_norm = [(s - b_min) / b_range for s in bm25_scores]

        # ---- merge into one score dict keyed by chunk text ----
        # using text as key because BM25 and vector store are separate indexes
        # and we don't have shared IDs to join on
        scores_by_text = {}

        for i, (text, metadata, _) in enumerate(vector_results):
            scores_by_text[text] = {
                "metadata": metadata,
                "vector_score": vector_norm[i],
                "bm25_score": 0.0,  # default — BM25 might not include this chunk
            }

        for i, (text, metadata, _) in enumerate(bm25_results):
            if text in scores_by_text:
                scores_by_text[text]["bm25_score"] = bm25_norm[i]
            else:
                # BM25 found something vector search didn't — include it
                scores_by_text[text] = {
                    "metadata": metadata,
                    "vector_score": 0.0,
                    "bm25_score": bm25_norm[i],
                }

        # ---- combine with weights ----
        combined = []
        for text, data in scores_by_text.items():
            final_score = (
                self.config.vector_weight * data["vector_score"]
                + self.config.bm25_weight * data["bm25_score"]
            )
            combined.append((text, data["metadata"], final_score))

        # sort best first, return top_k
        combined.sort(key=lambda x: x[2], reverse=True)
        return combined[:self.config.top_k]


    # FORMAT + LOG results

    def format_results(
        self,
        query: str,
        raw_results: list,
        elapsed: float,
        search_method: str,
    ) -> RetrievalArtifact:

        results = []

        for i, (text, metadata, score) in enumerate(raw_results):
            source = metadata.get("source", "unknown")
            source = Path(source).name if source != "unknown" else "unknown"

            result = RetrievalResult(
                content=text.strip(),
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
            search_method=search_method,
        )

    def retrieve(self, query: str, method: str = "hybrid") -> RetrievalArtifact:
        # method can be "vector", "bm25", or "hybrid"
        # default is hybrid — that's what the full pipeline will use
        self.logger.info(SEPARATOR_LINE)
        self.logger.info(f"Query  : {query}")
        self.logger.info(f"Method : {method}")

        start = time.time()

        if method == "vector":
            raw = self.vector_search(query)
        elif method == "bm25":
            raw = self.bm25_search(query)
        elif method == "hybrid":
            raw = self.hybrid_search(query)
        else:
            raise ValueError(f"Unknown search method: {method} — use vector, bm25, or hybrid")

        elapsed = time.time() - start
        artifact = self.format_results(query, raw, elapsed, method)

        self.logger.info(f"Retrieved {artifact.total_results} chunks in {elapsed:.3f}s")
        self.logger.info(SEPARATOR_LINE)

        return artifact

    def log_results(self, artifact: RetrievalArtifact) -> None:
        self.logger.info(f"Results for: '{artifact.query}' [{artifact.search_method}]")
        self.logger.info(SEPARATOR_LINE)

        for r in artifact.results:
            self.logger.info(f"[{r.chunk_index + 1}] {r.source} — score: {r.score}")
            self.logger.info(f"    {r.content[:150]}...")
            self.logger.info("")