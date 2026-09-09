from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)
from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.document_stores.in_memory import InMemoryDocumentStore


# ============================================================
# CONFIGURATION
# ============================================================

DOCUMENT_FOLDER = Path("documents")

QUESTIONS = [
    "What is the main topic discussed in the document?",
    "What problem is addressed?",
    "What methodology is used?",
    "What are the main objectives?",
    "What are the key findings?",
    "What advantages are mentioned?",
    "What limitations are discussed?",
    "What future work is suggested?",
    "What technologies are mentioned?",
    "What is the main conclusion?",
]


# ============================================================
# CHECK PDF FILES
# ============================================================

pdf_files = list(DOCUMENT_FOLDER.glob("*.pdf"))

if len(pdf_files) != 5:
    raise ValueError(
        f"Expected 5 PDF files, but found {len(pdf_files)}."
    )

print("=" * 70)
print("W7D1 - HAYSTACK PIPELINE ARCHITECTURE")
print("=" * 70)

print("\nPDF files found:")

for pdf in pdf_files:
    print(" -", pdf.name)


# ============================================================
# FILE CONVERTER + PREPROCESSING
# ============================================================

converter = PyPDFToDocument()

cleaner = DocumentCleaner()

splitter = DocumentSplitter(
    split_by="sentence",
    split_length=5,
    split_overlap=1,
)

converted = converter.run(
    sources=pdf_files
)

cleaned = cleaner.run(
    documents=converted["documents"]
)

split_result = splitter.run(
    documents=cleaned["documents"]
)

documents = split_result["documents"]

print("\nTotal document chunks:", len(documents))


# ============================================================
# BM25 PIPELINE
# ============================================================

print("\n" + "=" * 70)
print("BM25 RETRIEVAL")
print("=" * 70)

bm25_store = InMemoryDocumentStore()

bm25_store.write_documents(documents)

bm25_retriever = InMemoryBM25Retriever(
    document_store=bm25_store,
    top_k=3,
)

bm25_pipeline = Pipeline()

bm25_pipeline.add_component(
    "retriever",
    bm25_retriever
)


# ============================================================
# RUN 10 QUESTIONS WITH BM25
# ============================================================

bm25_scores = []

for number, question in enumerate(QUESTIONS, start=1):

    result = bm25_pipeline.run(
        {
            "retriever": {
                "query": question
            }
        }
    )

    retrieved = result["retriever"]["documents"]

    print(f"\nQ{number}: {question}")

    if retrieved:

        for rank, document in enumerate(
            retrieved,
            start=1
        ):

            print(
                f"  {rank}. "
                f"score={document.score:.4f}"
            )

            print(
                f"     source={document.meta.get('file_path', 'PDF')}"
            )

        bm25_scores.append(
            retrieved[0].score
        )

    else:

        print("  No documents retrieved.")


# ============================================================
# DENSE RETRIEVAL
# ============================================================

print("\n" + "=" * 70)
print("DENSE RETRIEVAL")
print("=" * 70)

dense_store = InMemoryDocumentStore(
    embedding_similarity_function="cosine"
)


document_embedder = SentenceTransformersDocumentEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

print("\nCreating document embeddings...")

document_embedder.warm_up()

embedded_documents = document_embedder.run(
    documents=documents
)

dense_store.write_documents(
    embedded_documents["documents"]
)


text_embedder = SentenceTransformersTextEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

text_embedder.warm_up()


dense_retriever = InMemoryEmbeddingRetriever(
    document_store=dense_store,
    top_k=3,
)


dense_pipeline = Pipeline()

dense_pipeline.add_component(
    "text_embedder",
    text_embedder
)

dense_pipeline.add_component(
    "retriever",
    dense_retriever
)

dense_pipeline.connect(
    "text_embedder.embedding",
    "retriever.query_embedding"
)


# ============================================================
# RUN SAME 10 QUESTIONS WITH DENSE RETRIEVAL
# ============================================================

dense_scores = []

for number, question in enumerate(QUESTIONS, start=1):

    result = dense_pipeline.run(
        {
            "text_embedder": {
                "text": question
            }
        }
    )

    retrieved = result["retriever"]["documents"]

    print(f"\nQ{number}: {question}")

    if retrieved:

        for rank, document in enumerate(
            retrieved,
            start=1
        ):

            print(
                f"  {rank}. "
                f"score={document.score:.4f}"
            )

            print(
                f"     source={document.meta.get('file_path', 'PDF')}"
            )

        dense_scores.append(
            retrieved[0].score
        )

    else:

        print("  No documents retrieved.")


# ============================================================
# COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("BM25 VS DENSE RETRIEVAL")
print("=" * 70)

if bm25_scores:

    print(
        "\nBM25 average top score:",
        round(
            sum(bm25_scores) / len(bm25_scores),
            4
        )
    )

if dense_scores:

    print(
        "Dense average top score:",
        round(
            sum(dense_scores) / len(dense_scores),
            4
        )
    )


print("\nManual evaluation scale:")
print("1 = Incorrect")
print("2 = Partially relevant")
print("3 = Highly relevant")

print("\nEvaluate the same 10 questions for both methods.")

print("\n" + "=" * 70)
print("W7D1 PIPELINE COMPLETED")
print("=" * 70)