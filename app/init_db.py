import os

from vector_store import build_vector_store, load_all_chunks
from hybrid_retriever import HybridRetriever


def build_all():
    print("=== Building Vector DB ===")
    vectorstore = build_vector_store()

    print("=== Loading all chunks ===")
    documents = load_all_chunks()

    print("=== Building Whoosh Index ===")
    # This triggers Whoosh index creation
    HybridRetriever(vectorstore=vectorstore, documents=documents)

    print("=== All initialization complete ===")
    