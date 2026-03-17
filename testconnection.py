# test_connections.py
# Run: python test_connections.py

import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 50)
print("TESTING ALL CONNECTIONS")
print("=" * 50)

# ─── TEST 1: Environment Variables ───────────────────
print("\n[1] Checking API keys in .env...")

groq_key = os.getenv("GROQ_API_KEY")
cohere_key = os.getenv("COHERE_API_KEY")
groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # default if not set

if groq_key:
    print(f"  ✅ GROQ_API_KEY found: {groq_key[:8]}...")
else:
    print("  ❌ GROQ_API_KEY not found — check your .env file")

if cohere_key:
    print(f"  ✅ COHERE_API_KEY found: {cohere_key[:8]}...")
else:
    print("  ❌ COHERE_API_KEY not found — check your .env file")

print(f"  ✅ GROQ_MODEL set to: {groq_model}")

# ─── TEST 2: Groq LLM ────────────────────────────────
print("\n[2] Testing Groq LLM connection...")

try:
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model=groq_model,     # Uses value from .env
        temperature=0,
        api_key=groq_key
    )

    response = llm.invoke("Say hello in one sentence.")
    print(f"  ✅ Groq works!")
    print(f"     Model: {groq_model}")
    print(f"     Response: {response.content}")

except Exception as e:
    print(f"  ❌ Groq failed: {e}")

# ─── TEST 3: HuggingFace Embeddings ──────────────────
print("\n[3] Testing HuggingFace Embeddings...")
print("     NOTE: First run downloads ~90MB model. Be patient.")

try:
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    test_embedding = embeddings.embed_query("test sentence")
    print(f"  ✅ HuggingFace Embeddings work!")
    print(f"     Embedding dimension: {len(test_embedding)}")

except Exception as e:
    print(f"  ❌ HuggingFace Embeddings failed: {e}")

# ─── TEST 4: Cohere Reranker ──────────────────────────
print("\n[4] Testing Cohere connection...")

try:
    import cohere

    co = cohere.Client(api_key=cohere_key)

    results = co.rerank(
        model="rerank-english-v3.0",
        query="How do I install ROS2?",
        documents=[
            "ROS2 installation requires Ubuntu 22.04",
            "Python is a programming language",
            "ROS2 Humble is installed using apt-get"
        ],
        top_n=2
    )

    print(f"  ✅ Cohere works!")
    print(f"     Top result index: {results.results[0].index}")

except Exception as e:
    print(f"  ❌ Cohere failed: {e}")

# ─── TEST 5: ChromaDB ─────────────────────────────────
print("\n[5] Testing ChromaDB...")

try:
    import chromadb

    client = chromadb.Client()
    collection = client.create_collection("test_collection")
    collection.add(
        documents=["This is a test document about ROS2"],
        ids=["test_id_1"]
    )
    results = collection.query(query_texts=["ROS2"], n_results=1)
    print(f"  ✅ ChromaDB works!")
    print(f"     Retrieved: {results['documents'][0][0]}")

except Exception as e:
    print(f"  ❌ ChromaDB failed: {e}")

# ─── SUMMARY ──────────────────────────────────────────
print("\n" + "=" * 50)
print("TEST COMPLETE")
print("If all show ✅ you are ready to start coding!")
print("=" * 50)