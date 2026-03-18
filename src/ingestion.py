import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# PDF Loading
from langchain_community.document_loaders import PyPDFLoader

#P Text Splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Hugging Embeddings
from langchain_core import documents
from langchain_huggingface import HuggingFaceEmbeddings

# ChromaDB Vector Store
from langchain_community.vectorstores import Chroma


load_dotenv()

DOCUMENTS_PATH = Path("data/documents")
CHROMA_DB_PATH = Path("chroma_db")


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

COLLECTION_NAME= "research_papers"



# STEP 1: LOAD PDFs
def load_documents(documents_path:Path)->list:
    """
    Load all the pdf and return a list of LangChain Document Objects.
    Each document has: page_content + metadata (source,page)
    """
    all_documents=[]
    pdf_files = list(documents_path.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {documents_path}")
        sys.exit(1)
    
    print(f"File Found { len(pdf_files)} PDF files")
    print("-"*40)

    for pdf_path in sorted(pdf_files):
        try:
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            all_documents.extend(documents)
            print(f"Loaded: {pdf_path.name} ({len(documents)} papers)")
        except Exception as e:
            print(f" Failed to load {pdf_path.name}: {e}")
        
    print("-"*40)
    print(f" Total pages loaded: {len(all_documents)}")
    return all_documents



# STEP 2: SPLIT INTO CHUNKS
def split_documents(documents:list)->list:
    """
    Split Documents into smaller chunks while keeping metadata.
    This metada = citations later!
    """
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap= CHUNK_OVERLAP,
        length_function = len,
        separators = ["\n\n","\n"," ",""] # tries to split at paragraphs first 
    )

    chunks = splitter.split_documents(documents)

    #Clean up the source path in metadata - keep only filename
    for chunk in chunks:
        source = chunk.metadata.get("source","")
        chunk.metadata["source"] = Path(source).name

    print(f"\n Chunking complete:")
    print(f" Total chunks creaetd: {len(chunks)}")
    print(f" Chunk size: { CHUNK_SIZE} chars")
    print(f" Chunk overlap: { CHUNK_OVERLAP} chars")
    return chunks



# STEP 3: CREATE EMBEDINGS + STORE IN CHROMADB

def create_vector_store(chunks:list)->Chroma:
    """
    Convert chunks to embeddings using HuggingFace.
    Store everything in ChromaDB locally.
    """
    print(f"\n Loading embedding model: { EMBEDDING_MODEL}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL,
        model_kwargs={"device":"cpu"},
        encode_kwargs={"normalize_embeddings":True}
    )

    print(" Embedding model loaded")
    print(f"\n Stroing chunks in ChromaDB....")
    print(f" Locataion: {CHROMA_DB_PATH}")
    print(f" Collection: {COLLECTION_NAME}")
    
    # This automatically converts every chunks to embeddings and stores them in local ChromaDB
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding = embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory = str(CHROMA_DB_PATH)
    )
    print(f" Vectore store created successfully")
    return vector_store

 

# STEP 4: VERIFY EVERYTHING WORKED

def verify_vector_store(vector_store:Chroma):
    """
    Run a quick test serach to confirm everythin works.
    """
    print(f"\n Running verification test....")

    test_query = "What is the attention mechanism in tranformers?"
    results = vector_store.similarity_search(test_query,k=3)

    print(f" Test query: '{test_query}")
    print(f" Results found: {len(results)}")
    print("\n Top result preview:")
    print(" " + "-"*36)

    if results:
        top =results[0]
        preview = top.page_content[:200].replace("\n"," ")
        source = top.netadata.get("source","unknown")
        page = top.metadata.get("page","?")
        print(f" Source: {source}")
        print(f" Page: { page}")
        print(f" Text: { preview}...")

    print(" "+ "-"*36)
    print(" Verification passed!")


# MAIN 


def main():
    print("=" * 50)
    print("   ASK MY DOCS — Document Ingestion Pipeline")
    print("=" * 50)

    # Check documents folder exists
    if not DOCUMENTS_PATH.exists():
        print(f"❌ Documents folder not found: {DOCUMENTS_PATH}")
        sys.exit(1)

    # Step 1: Load
    print("\n📖 STEP 1: Loading PDFs...")
    documents = load_documents(DOCUMENTS_PATH)

    # Step 2: Chunk
    print("\n✂️  STEP 2: Splitting into chunks...")
    chunks = split_documents(documents)

    # Step 3: Embed + Store
    print("\n🔢 STEP 3: Creating embeddings + storing in ChromaDB...")
    vector_store = create_vector_store(chunks)

    # Step 4: Verify
    print("\n✅ STEP 4: Verifying...")
    verify_vector_store(vector_store)

    # Done!
    print("\n" + "=" * 50)
    print("🎉 INGESTION COMPLETE!")
    print(f"   Documents loaded : {len(documents)} pages")
    print(f"   Chunks created   : {len(chunks)}")
    print(f"   ChromaDB saved   : {CHROMA_DB_PATH}/")
    print("=" * 50)
    print("\n✅ Ready for Day 3 — Building retrieval.py")


if __name__ == "__main__":
    main()

        