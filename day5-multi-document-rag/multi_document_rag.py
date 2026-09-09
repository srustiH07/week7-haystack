import os
import time
import chromadb

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

print("=" * 70)
print("W7D5 - MULTI-DOCUMENT RAG SYSTEM")
print("=" * 70)

OLLAMA_HOST = "http://localhost:11434"

Settings.llm = Ollama(
    model="llama3.2:3b",
    base_url=OLLAMA_HOST,
    request_timeout=120.0,
    context_window=2048
)

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text:latest",
    base_url=OLLAMA_HOST
)

print("\nOllama configured.")

documents = SimpleDirectoryReader("documents").load_data()

print(f"\nDocuments loaded: {len(documents)}")

for document in documents:
    path = document.metadata.get("file_path", "unknown")
    print("-", os.path.basename(path))

print("\nCreating ChromaDB vector store...")

chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="w7d5_multidocument_rag"
)

vector_store = ChromaVectorStore(
    chroma_collection=collection
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

print("Creating vector index...")

start = time.perf_counter()

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

index_time = time.perf_counter() - start

print(f"Index created in {index_time:.2f} seconds.")

query_engine = index.as_query_engine(similarity_top_k=3)

print("\nQuery engine created successfully.")

questions = [
    "What is artificial intelligence?",
    "What are the main types of machine learning?",
    "What are large language models used for?",
    "What is Retrieval-Augmented Generation?",
    "What is ChromaDB?"
]

results = []

print("\n" + "=" * 70)
print("MULTI-DOCUMENT RAG QUERY TEST")
print("=" * 70)

for number, question in enumerate(questions, 1):

    print("\n" + "-" * 70)
    print(f"Question {number}: {question}")
    print("-" * 70)

    try:
        start = time.perf_counter()

        response = query_engine.query(question)

        latency = time.perf_counter() - start

        print("\nAnswer:")
        print(response)

        print(f"\nLatency: {latency:.2f} seconds")

        sources = set()

        for node in response.source_nodes:
            path = node.node.metadata.get("file_path", "unknown")
            sources.add(os.path.basename(path))

        print("\nSources:")

        for source in sorted(sources):
            print("-", source)

        results.append({
            "question": question,
            "answer": str(response),
            "latency": latency,
            "sources": sorted(sources)
        })

    except Exception as error:

        print("\nError:", error)

        results.append({
            "question": question,
            "answer": "ERROR: " + str(error),
            "latency": 0,
            "sources": []
        })

os.makedirs("results", exist_ok=True)

with open("results/rag_results.txt", "w", encoding="utf-8") as file:

    file.write("W7D5 - MULTI-DOCUMENT RAG RESULTS\n")
    file.write("=" * 70 + "\n\n")

    file.write(f"Documents indexed: {len(documents)}\n")
    file.write(f"Index creation time: {index_time:.2f} seconds\n\n")

    for number, result in enumerate(results, 1):

        file.write(f"Question {number}: {result['question']}\n")
        file.write(f"Answer: {result['answer']}\n")
        file.write(f"Latency: {result['latency']:.2f} seconds\n")
        file.write(f"Sources: {', '.join(result['sources'])}\n")
        file.write("-" * 70 + "\n")

print("\nResults saved to:")
print("results/rag_results.txt")

print("\n" + "=" * 70)
print("W7D5 MULTI-DOCUMENT RAG SYSTEM COMPLETED")
print("=" * 70)