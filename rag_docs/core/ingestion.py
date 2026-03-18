# rag_docs/core/ingestion.py
# Handles everything related to loading PDFs and storing them in ChromaDB
# Run this once to set up the knowledge base, then use retrieval.py to search

import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_docs.entity import IngestionConfig, IngestionArtifact
from rag_docs.logging.logger import get_logger
from rag_docs.utils.file_utils import folder_is_empty, delete_folder, ask_user_yes_no
from rag_docs.config.settings import LOGS_PATH, SEPARATOR_LINE


class DocumentIngestion:

    def __init__(self, config: IngestionConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.logger.info(f"Starting ingestion — docs: {config.documents_path}")


    # ------------------------------------------------------------------
    # STEP 0 — make sure we don't accidentally double-ingest
    # ------------------------------------------------------------------

    def check_existing_db(self) -> bool:
        # Returns True if we should SKIP ingestion (DB already good)
        # Returns False if we should proceed (no DB or user said delete)
        try:
            if folder_is_empty(self.config.chroma_db_path):
                return False  # nothing there, just proceed

            self.logger.warning("ChromaDB folder already exists!")
            self.logger.warning("Re-running ingestion will create duplicate chunks.")

            if ask_user_yes_no("Delete existing DB and start fresh?"):
                deleted = delete_folder(self.config.chroma_db_path)
                if deleted:
                    self.logger.info("Deleted old DB, starting fresh.")
                    return False
                else:
                    self.logger.error("Couldn't delete DB folder, something is wrong.")
                    sys.exit(1)

            # user said no — keep existing DB
            self.logger.info("Keeping existing DB, skipping ingestion.")
            return True

        except Exception as e:
            self.logger.error(f"check_existing_db crashed: {e}")
            raise


    # ------------------------------------------------------------------
    # STEP 1 — read all the PDF files
    # ------------------------------------------------------------------

    def load_documents(self) -> list:
        try:
            all_documents = []
            failed = []

            pdf_files = list(self.config.documents_path.glob("*.pdf"))

            if not pdf_files:
                self.logger.error(f"No PDF files found in {self.config.documents_path}")
                self.logger.error("Add your PDFs to data/documents/ and try again.")
                sys.exit(1)

            self.logger.info(f"Found {len(pdf_files)} PDFs, loading them now...")
            print(SEPARATOR_LINE)

            for pdf_path in sorted(pdf_files):
                try:
                    loader = PyPDFLoader(str(pdf_path))
                    pages = loader.load()

                    if not pages:
                        self.logger.warning(f"Skipping {pdf_path.name} — came back empty")
                        continue

                    all_documents.extend(pages)
                    self.logger.info(f"  {pdf_path.name:<40} {len(pages):>3} pages")

                except Exception as e:
                    failed.append(pdf_path.name)
                    self.logger.error(f"  Failed to load {pdf_path.name}")
                    self.logger.debug(f"  Error detail: {e}")

            print(SEPARATOR_LINE)
            self.logger.info(f"Loaded {len(all_documents)} pages total")
            self.logger.info(f"Success: {len(pdf_files) - len(failed)}/{len(pdf_files)} files")

            if failed:
                self.logger.warning(f"These files failed: {failed}")

            return all_documents

        except Exception as e:
            self.logger.error(f"load_documents crashed: {e}")
            raise


    # ------------------------------------------------------------------
    # STEP 2 — split pages into chunks the embedding model can handle
    # ------------------------------------------------------------------

    def split_documents(self, documents: list) -> list:
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                length_function=len,
                # tries paragraph breaks first, then lines, then words
                separators=["\n\n", "\n", " ", ""]
            )

            raw_chunks = splitter.split_documents(documents)
            self.logger.info(f"Got {len(raw_chunks)} raw chunks before filtering")

            good_chunks = []
            skipped = 0

            for chunk in raw_chunks:
                text = chunk.page_content.strip()

                # skip if too short to be useful
                if len(text) < self.config.min_chunk_length:
                    skipped += 1
                    continue

                # skip if it's just numbers and symbols (page numbers, tables, etc.)
                if not any(c.isalpha() for c in text):
                    skipped += 1
                    continue

                # clean up source path — we only need the filename, not full path
                chunk.metadata["source"] = Path(chunk.metadata.get("source", "")).name
                good_chunks.append(chunk)

            self.logger.info(f"After filtering — kept {len(good_chunks)}, skipped {skipped}")

            # show a breakdown of chunks per document
            counts = {}
            for chunk in good_chunks:
                src = chunk.metadata.get("source", "unknown")
                counts[src] = counts.get(src, 0) + 1

            print(f"\n   Chunks per document:")
            print(f"   {'Document':<40} {'Chunks':>6}")
            print(f"   {'─'*40} {'─'*6}")
            for name, count in sorted(counts.items()):
                print(f"   {name:<40} {count:>6}")
            print(f"   {'─'*40} {'─'*6}")

            return good_chunks

        except Exception as e:
            self.logger.error(f"split_documents crashed: {e}")
            raise


    # ------------------------------------------------------------------
    # STEP 3 — convert chunks to vectors and store in ChromaDB
    # ------------------------------------------------------------------

    def create_vector_store(self, chunks: list) -> Chroma:
        try:
            # load the embedding model — first time downloads ~90MB, then cached
            self.logger.info(f"Loading embedding model: {self.config.embedding_model}")
            self.logger.info("(first run downloads ~90MB, after that it's instant)")

            embedding_model = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={"device": self.config.embedding_device},
                encode_kwargs={"normalize_embeddings": True}
            )
            self.logger.info("Embedding model loaded, starting to store chunks...")

            # process in batches so we can see progress and avoid memory issues
            vector_store = None
            batch_size = self.config.batch_size
            total = len(chunks)
            total_batches = (total + batch_size - 1) // batch_size

            for batch_num, start in enumerate(range(0, total, batch_size), 1):
                end = min(start + batch_size, total)
                batch = chunks[start:end]
                pct = (end / total) * 100

                self.logger.info(f"Batch {batch_num}/{total_batches} — chunks {start+1} to {end} ({pct:.0f}%)")

                if vector_store is None:
                    # first batch creates the collection
                    vector_store = Chroma.from_documents(
                        documents=batch,
                        embedding=embedding_model,
                        collection_name=self.config.collection_name,
                        persist_directory=str(self.config.chroma_db_path)
                    )
                else:
                    # rest of the batches just add to it
                    vector_store.add_documents(batch)

            self.logger.info(f"Done! Stored {total} chunks in ChromaDB.")
            return vector_store

        except Exception as e:
            self.logger.error(f"create_vector_store crashed: {e}")
            raise


    # ------------------------------------------------------------------
    # STEP 4 — quick sanity check to make sure search actually works
    # ------------------------------------------------------------------

    def verify_vector_store(self, vector_store: Chroma) -> bool:
        try:
            self.logger.info("Running a few test searches to verify everything works...")

            # these topics are all covered in the 28 documents
            test_queries = [
                "What is the attention mechanism in transformers?",
                "How does backpropagation work in neural networks?",
                "What is LoRA fine-tuning?",
            ]

            passed = 0

            for i, query in enumerate(test_queries, 1):
                try:
                    results = vector_store.similarity_search(query, k=3)

                    if not results:
                        self.logger.warning(f"Test {i} got zero results for: {query}")
                        continue

                    best = results[0]
                    source = best.metadata.get("source", "unknown")
                    page = best.metadata.get("page", "?")
                    preview = best.page_content[:120].replace("\n", " ")

                    self.logger.info(f"Test {i} OK — {source} page {page}")
                    self.logger.info(f"  > {preview}")
                    passed += 1

                except Exception as e:
                    self.logger.error(f"Test {i} threw an error: {e}")

            print(f"\n   {passed}/{len(test_queries)} verification tests passed")

            if passed == len(test_queries):
                self.logger.info("All tests passed, ingestion looks good!")
                return True

            self.logger.warning("Some tests failed — check logs/rag_docs.log for details")
            return False

        except Exception as e:
            self.logger.error(f"verify_vector_store crashed: {e}")
            raise


    # ------------------------------------------------------------------
    # main entry point — called from main.py, runs everything in order
    # ------------------------------------------------------------------

    def initiate_ingestion(self) -> IngestionArtifact:
        try:
            # step 0 — check if we already have a database
            self.logger.info("Checking if ChromaDB already exists...")
            db_already_exists = self.check_existing_db()

            if db_already_exists:
                # nothing to do, return an artifact showing we skipped
                return IngestionArtifact(
                    chroma_db_path=self.config.chroma_db_path,
                    collection_name=self.config.collection_name,
                    total_pages_loaded=0,
                    total_chunks_stored=0,
                    total_files_processed=0,
                    verification_passed=True,
                    log_file_path=LOGS_PATH / "rag_docs.log"
                )

            # step 1 — load PDFs
            self.logger.info("Step 1: Loading PDFs...")
            documents = self.load_documents()

            # step 2 — split into chunks
            self.logger.info("Step 2: Splitting into chunks...")
            chunks = self.split_documents(documents)

            if not chunks:
                self.logger.error("No valid chunks came out of splitting — check your PDFs")
                sys.exit(1)

            # step 3 — embed and store
            self.logger.info("Step 3: Embedding and storing in ChromaDB...")
            vector_store = self.create_vector_store(chunks)

            # step 4 — verify
            self.logger.info("Step 4: Verifying...")
            all_good = self.verify_vector_store(vector_store)

            # build the result summary and return it
            artifact = IngestionArtifact(
                chroma_db_path=self.config.chroma_db_path,
                collection_name=self.config.collection_name,
                total_pages_loaded=len(documents),
                total_chunks_stored=len(chunks),
                total_files_processed=len(list(self.config.documents_path.glob("*.pdf"))),
                verification_passed=all_good,
                log_file_path=LOGS_PATH / "rag_docs.log"
            )

            self.logger.info("Ingestion finished successfully.")
            return artifact

        except Exception as e:
            self.logger.error(f"Ingestion pipeline failed: {e}")
            raise