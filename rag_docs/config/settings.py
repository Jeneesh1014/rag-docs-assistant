# ALL configuration in ONE place
# Change settings here — everything else updates automatically

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base directory = the project root folder
BASE_DIR       = Path(__file__).resolve().parent.parent.parent

DOCUMENTS_PATH = BASE_DIR / "data" / "documents"
CHROMA_DB_PATH = BASE_DIR / "chroma_db"
LOGS_PATH      = BASE_DIR / "logs"

# ── Chunking Settings
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 50
MIN_CHUNK_LENGTH = 50

# ── Embedding Model
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"

# ── ChromaDB Settings
COLLECTION_NAME = "research_papers"
BATCH_SIZE      = 500

# ── Display Settings
SEPARATOR_LINE = "─" * 50

# ── Retrieval Settings
TOP_K         = 8
VECTOR_WEIGHT = 0.6
BM25_WEIGHT   = 0.4

# ── Reranking Settings
COHERE_MODEL = "rerank-english-v3.0"
RERANK_TOP_N = 3

# ── Generation Settings
GROQ_MODEL  = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_TOKENS  = 1024
TEMPERATURE = 0.1