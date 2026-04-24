# rag_engine.py
from vector_store import load_vector_store
from qa_chain import get_qa_chain

_vectorstore = None
_qa_chain_cache = {}  # Cache for different configurations (hybrid + k combinations)


def get_dynamic_k(question: str) -> int:
    """
    Dynamically adjust retrieval count based on question complexity.
    This improves answer quality without wasting tokens.
    """
    q_lower = question.lower()
    
    # Summary / overview questions need more context
    summary_keywords = ["summarize", "overview", "what were the main", "list all", 
                        "all decisions", "all action items", "give me a summary",
                        "what happened", "key points", "main topics"]
    if any(keyword in q_lower for keyword in summary_keywords):
        return 10
    
    # Complex analytical questions
    analytical_keywords = ["compare", "contrast", "arguments", "perspectives", 
                          "concerns", "pros and cons", "different views",
                          "why did", "how did", "analysis"]
    if any(keyword in q_lower for keyword in analytical_keywords):
        return 8
    
    # Specific factual questions (needs precision, less noise)
    factual_keywords = ["what is", "what was", "who said", "when did", "where did",
                        "how much", "what time", "specific", "exact"]
    if any(keyword in q_lower for keyword in factual_keywords):
        return 4
    
    # Questions with /meeting tag (user wants exhaustive search)
    if q_lower.startswith("/meeting"):
        return 10
    
    # Default for general questions
    return 6


def get_cache_key(hybrid: bool, bm25_weight: float, k: int) -> str:
    """Generate cache key for different configurations."""
    return f"hybrid_{hybrid}_bm25_{bm25_weight}_k_{k}"


def get_qa_chain_singleton(hybrid=True, bm25_weight=0.3, k=6):
    """
    Get or create QA chain with caching for different configurations.
    
    Args:
        hybrid: Use hybrid search (BM25 + semantic)
        bm25_weight: Weight for BM25 (0-1). Semantic = 1 - bm25_weight.
        k: Number of documents to retrieve
    """
    global _vectorstore, _qa_chain_cache
    
    cache_key = get_cache_key(hybrid, bm25_weight, k)
    
    # Return cached chain if exists
    if cache_key in _qa_chain_cache:
        print(f" Using cached QA chain (k={k}, hybrid={hybrid}, bm25={bm25_weight})")
        return _qa_chain_cache[cache_key]
    
    # Load vector store (only once)
    if _vectorstore is None:
        print(" Loading vector store...")
        _vectorstore = load_vector_store()
    
    # Create new chain with specified parameters
    print(f"🔧 Creating new QA chain (k={k}, hybrid={hybrid}, bm25={bm25_weight})")
    qa_chain, _ = get_qa_chain(
        _vectorstore,
        k=k,
        hybrid=hybrid,
        bm25_weight=bm25_weight
    )
    
    # Cache it
    _qa_chain_cache[cache_key] = qa_chain
    return qa_chain


def get_rag_answer(question: str, hybrid=True, bm25_weight=0.3, k=None):
    """
    Get RAG answer for a question.
    
    Args:
        question: User's question
        hybrid: Use hybrid search
        bm25_weight: BM25 weight (only used if hybrid=True)
        k: Number of documents to retrieve (auto-calculated if None)
    """
    # Auto-calculate k if not provided
    if k is None:
        k = get_dynamic_k(question)
        print(f" Dynamic k = {k} for question: '{question[:50]}...'")
    
    qa = get_qa_chain_singleton(hybrid=hybrid, bm25_weight=bm25_weight, k=k)
    response = qa.invoke(question)
    
    return {
        "answer": response["result"],
        "source_documents": response["source_documents"],
        "k_used": k  # Optional: include in response for debugging
    }