import os
from pathlib import Path

from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)

BASE_DIR = Path(__file__).parent
PDF_DIR = BASE_DIR / "documents"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("W7D2 - HAYSTACK RETRIEVAL: BM25 & DENSE RETRIEVAL")
print("=" * 70)

# ---------------------------------------------------------
# 1. Find PDF files
# ---------------------------------------------------------

pdf_files = sorted(PDF_DIR.glob("*.pdf"))

print("\nPDF files found:")

for pdf in pdf_files:
    print(" -", pdf.name)

if len(pdf_files) != 5:
    print(f"\nWARNING: Expected 5 PDFs, found {len(pdf_files)}.")

# ---------------------------------------------------------
# 2. Convert PDFs
# ---------------------------------------------------------

converter = PyPDFToDocument()

documents = []

for pdf in pdf_files:
    result = converter.run(sources=[str(pdf)])
    documents.extend(result["documents"])

print(f"\nDocuments converted: {len(documents)}")

# ---------------------------------------------------------
# 3. Clean and split documents
# ---------------------------------------------------------

cleaner = DocumentCleaner()

cleaned = cleaner.run(documents=documents)["documents"]

splitter = DocumentSplitter(
    split_by="word",
    split_length=200,
    split_overlap=20
)

split_documents = splitter.run(
    documents=cleaned
)["documents"]

print(f"Document chunks created: {len(split_documents)}")

# ---------------------------------------------------------
# 4. BM25 Retrieval
# ---------------------------------------------------------

document_store_bm25 = InMemoryDocumentStore()

document_store_bm25.write_documents(split_documents)

bm25_retriever = InMemoryBM25Retriever(
    document_store=document_store_bm25
)

# ---------------------------------------------------------
# 5. Dense Retrieval
# ---------------------------------------------------------

print("\nLoading dense embedding model...")

document_embedder = SentenceTransformersDocumentEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

document_embedder.warm_up()

embedded_documents = document_embedder.run(
    documents=split_documents
)["documents"]

document_store_dense = InMemoryDocumentStore()

document_store_dense.write_documents(embedded_documents)

text_embedder = SentenceTransformersTextEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

text_embedder.warm_up()

# ---------------------------------------------------------
# 6. Questions
# ---------------------------------------------------------

questions = [
    "What is the main topic of the document?",
    "What are the important concepts discussed?",
    "What are the key advantages mentioned?",
    "What problems are discussed in the documents?",
    "What methods or techniques are explained?",
    "What are the main applications mentioned?",
    "What challenges are identified?",
    "What are the important results or findings?",
    "What recommendations are provided?",
    "What is the overall conclusion?"
]

# ---------------------------------------------------------
# 7. Run BM25 retrieval
# ---------------------------------------------------------

bm25_results = []

print("\n" + "=" * 70)
print("BM25 RETRIEVAL")
print("=" * 70)

for i, question in enumerate(questions, start=1):

    result = bm25_retriever.run(
        query=question,
        top_k=3
    )

    docs = result["documents"]

    bm25_results.append(docs)

    print(f"\nQuestion {i}: {question}")

    for rank, doc in enumerate(docs, start=1):
        text = doc.content.replace("\n", " ")[:180]
        print(f"  {rank}. {text}...")

# ---------------------------------------------------------
# 8. Run Dense retrieval
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("DENSE RETRIEVAL")
print("=" * 70)

dense_results = []

for i, question in enumerate(questions, start=1):

    query_embedding = text_embedder.run(
        text=question
    )["embedding"]

    result = document_store_dense.embedding_retrieval(
        query_embedding=query_embedding,
        top_k=3
    )

    docs = result

    dense_results.append(docs)

    print(f"\nQuestion {i}: {question}")

    for rank, doc in enumerate(docs, start=1):
        text = doc.content.replace("\n", " ")[:180]
        print(f"  {rank}. {text}...")

# ---------------------------------------------------------
# 9. Manual evaluation template
# ---------------------------------------------------------

evaluation_file = RESULTS_DIR / "manual_evaluation.txt"

with open(evaluation_file, "w", encoding="utf-8") as f:

    f.write("W7D2 - BM25 vs Dense Retrieval Manual Evaluation\n")
    f.write("=" * 60 + "\n\n")

    f.write("Rating:\n")
    f.write("1 = Incorrect\n")
    f.write("2 = Partially relevant\n")
    f.write("3 = Highly relevant\n\n")

    for i, question in enumerate(questions, start=1):

        f.write(f"Question {i}: {question}\n")
        f.write("BM25 rating: \n")
        f.write("Dense rating: \n")
        f.write("Comments: \n\n")

print("\n" + "=" * 70)
print("W7D2 PIPELINE COMPLETED")
print("=" * 70)

print("\nManual evaluation file created:")
print(evaluation_file)