# rag_engine.py
from vector_store import load_vector_store
from qa_chain import get_qa_chain

# Singletons
_vectorstore = None
_qa_chain_semantic = None   # pure semantic chain (reused for all k values dynamically)
_qa_chain_hybrid = None     # hybrid chain (reused for all k values dynamically)

# We build two chains (semantic / hybrid) and pass k at query time via
# the retriever's `search_kwargs`.  For HybridRetriever we reset `k` directly.


def get_dynamic_k(question: str) -> int:
    q_lower = question.lower()
    summary_kw = ["summarize", "overview", "what were the main", "list all",
                  "all decisions", "all action items", "give me a summary",
                  "what happened", "key points", "main topics"]
    if any(kw in q_lower for kw in summary_kw):
        return 10

    analytical_kw = ["compare", "contrast", "arguments", "perspectives",
                     "concerns", "pros and cons", "different views",
                     "why did", "how did", "analysis"]
    if any(kw in q_lower for kw in analytical_kw):
        return 8

    factual_kw = ["what is", "what was", "who said", "when did", "where did",
                  "how much", "what time", "specific", "exact"]
    if any(kw in q_lower for kw in factual_kw):
        return 4

    return 6


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        print("Loading vector store...")
        _vectorstore = load_vector_store()
    return _vectorstore


def get_rag_answer(question: str, hybrid: bool = True, bm25_weight: float = 0.3, k: int = None):
    """
    Answer a question using the RAG pipeline.

    Args:
        question:    User question.
        hybrid:      Use hybrid (BM25 + semantic) retrieval.
        bm25_weight: BM25 weight for hybrid mode.
        k:           Docs to retrieve (auto if None).
    """
    if k is None:
        k = get_dynamic_k(question)
        print(f"Dynamic k={k} for: '{question[:60]}...'")

    vs = _get_vectorstore()

    # Build a fresh chain only if configuration changed.
    # Rebuilding is fast (no embedding re-load); the vector store and Whoosh index are reused.
    qa_chain, _ = get_qa_chain(vs, k=k, hybrid=hybrid, bm25_weight=bm25_weight)

    response = qa_chain.invoke(question)
    return {
        "answer": response["result"],
        "source_documents": response.get("source_documents", []),
        "k_used": k,
    }
