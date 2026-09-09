import os
import time

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb


print("=" * 70)
print("W7D3 - LLAMAINDEX DOCUMENT INDEXING & QUERYING")
print("=" * 70)


# ============================================================
# 1. Configure Ollama LLM and Embeddings
# ============================================================

Settings.llm = Ollama(
    model="llama3.2:3b",
    request_timeout=120.0,
    context_window=2048,
    additional_kwargs={
        "num_ctx": 2048
    }
)


Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434"
)

print("\nOllama LLM and embedding model configured.")


# ============================================================
# 2. Load the 5 text documents
# ============================================================

DOCUMENT_FOLDER = "documents"

documents = SimpleDirectoryReader(
    DOCUMENT_FOLDER
).load_data()

print(f"\nDocuments loaded: {len(documents)}")

for document in documents:
    print(
        "-",
        os.path.basename(
            document.metadata.get("file_name", "unknown")
        )
    )


# ============================================================
# 3. Create LlamaIndex VectorStoreIndex
# ============================================================

print("\nCreating LlamaIndex VectorStoreIndex...")

start_time = time.perf_counter()

index = VectorStoreIndex.from_documents(documents)

build_time = time.perf_counter() - start_time

print(
    f"LlamaIndex index created in {build_time:.2f} seconds."
)


# ============================================================
# 4. Create Query Engine
# ============================================================

query_engine = index.as_query_engine(
    similarity_top_k=2
)


# ============================================================
# 5. Ten Questions
# ============================================================

queries = [
    "What is artificial intelligence?",
    "What are the main types of machine learning?",
    "What is supervised learning?",
    "What is Retrieval-Augmented Generation?",
    "Why is RAG useful?",
    "What are large language models?",
    "What architecture do modern LLMs use?",
    "What is ChromaDB?",
    "What are embeddings?",
    "How are vector databases used in RAG?"
]


# ============================================================
# 6. Run 10 Queries with LlamaIndex
# ============================================================

print("\n" + "=" * 70)
print("LLAMAINDEX - 10 QUERY RESULTS")
print("=" * 70)

llamaindex_latencies = []

for number, query in enumerate(queries, start=1):

    start_time = time.perf_counter()

    response = query_engine.query(query)

    latency = time.perf_counter() - start_time

    llamaindex_latencies.append(latency)

    print(f"\nQuestion {number}: {query}")
    print("-" * 70)

    print("Answer:")
    print(response)

    print("\nSource documents:")

    source_files = set()

    for node in response.source_nodes:
        file_name = node.node.metadata.get(
            "file_name",
            "unknown"
        )

        source_files.add(
            os.path.basename(file_name)
        )

    for file_name in source_files:
        print("-", file_name)

    print(f"Latency: {latency:.2f} seconds")


# ============================================================
# 7. Create ChromaDB Vector Store
# ============================================================

print("\n" + "=" * 70)
print("CONNECTING LLAMAINDEX TO CHROMADB")
print("=" * 70)

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="w7d3_documents"
)

chroma_vector_store = ChromaVectorStore(
    chroma_collection=collection
)


# ============================================================
# 8. Create ChromaDB-backed LlamaIndex
# ============================================================

print("\nCreating ChromaDB-backed index...")

chroma_start = time.perf_counter()

chroma_index = VectorStoreIndex.from_documents(
    documents,
    vector_store=chroma_vector_store
)

chroma_build_time = (
    time.perf_counter() - chroma_start
)

print(
    f"ChromaDB index created in "
    f"{chroma_build_time:.2f} seconds."
)


# ============================================================
# 9. ChromaDB Query Engine
# ============================================================

chroma_query_engine = chroma_index.as_query_engine(
    similarity_top_k=2
)


# ============================================================
# 10. Run the Same 10 Queries Again
# ============================================================

print("\n" + "=" * 70)
print("CHROMADB - 10 QUERY RESULTS")
print("=" * 70)

chroma_latencies = []

for number, query in enumerate(queries, start=1):

    start_time = time.perf_counter()

    response = chroma_query_engine.query(query)

    latency = time.perf_counter() - start_time

    chroma_latencies.append(latency)

    print(f"\nQuestion {number}: {query}")
    print("-" * 70)

    print("Answer:")
    print(response)

    print("\nSource documents:")

    source_files = set()

    for node in response.source_nodes:
        file_name = node.node.metadata.get(
            "file_name",
            "unknown"
        )

        source_files.add(
            os.path.basename(file_name)
        )

    for file_name in source_files:
        print("-", file_name)

    print(f"Latency: {latency:.2f} seconds")


# ============================================================
# 11. Compare Latency
# ============================================================

average_llamaindex_latency = (
    sum(llamaindex_latencies)
    / len(llamaindex_latencies)
)

average_chroma_latency = (
    sum(chroma_latencies)
    / len(chroma_latencies)
)


print("\n" + "=" * 70)
print("LATENCY COMPARISON")
print("=" * 70)

print(
    f"LlamaIndex average query latency: "
    f"{average_llamaindex_latency:.2f} seconds"
)

print(
    f"ChromaDB average query latency: "
    f"{average_chroma_latency:.2f} seconds"
)


# ============================================================
# 12. Completion
# ============================================================

print("\n" + "=" * 70)
print("W7D3 PIPELINE COMPLETED")
print("=" * 70)