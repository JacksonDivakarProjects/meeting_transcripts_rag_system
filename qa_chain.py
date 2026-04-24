# qa_chain.py
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
from config import GROQ_API_KEY, GROQ_MODEL
import os

# Import hybrid retriever (create this file)
from hybrid_retriever import HybridRetriever
from vector_store import load_all_chunks

class UniqueDocsRetriever(BaseRetriever):
    vectorstore: any
    k: int = 5

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=self.k * 3)
        seen = set()
        unique = []
        for doc, _ in docs_with_scores:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                # Inject metadata into the page_content as a prefix
                meta = doc.metadata
                speaker = meta.get("speaker", "unknown")
                timestamp = meta.get("timestamp_str", "00:00")
                source = meta.get("source_file", "unknown")
                prefix = f"[Speaker: {speaker}, Time: {timestamp}, File: {source}] "
                new_content = prefix + doc.page_content
                # Create a new Document with the prefixed content, keep original metadata
                new_doc = Document(page_content=new_content, metadata=meta)
                unique.append(new_doc)
            if len(unique) >= self.k:
                break
        return unique

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)

    async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return self._get_relevant_documents(query)


def get_qa_chain(vectorstore, k=5, hybrid=False, bm25_weight=0.3):
    """
    Returns a QA chain with optional hybrid search.
    
    Args:
        vectorstore: Chroma vector store
        k: Number of documents to retrieve
        hybrid: If True, use HybridRetriever (BM25 + semantic)
        bm25_weight: Weight for BM25 (0-1). Semantic weight = 1 - bm25_weight.
    """
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.0, max_retries=2, timeout=60)
    
    prompt = PromptTemplate(
    template="""You are a meeting analyst. Answer the user's question using ONLY the provided context.

Each piece of context starts with a citation number in brackets, e.g., [1], [2], etc.
You MUST refer to those numbers when citing a source. For example: "According to [1], the budget was increased."

If the context does not contain the answer, say "I cannot find that information."

Context:
{context}

Question: {question}

Answer (with citation numbers in brackets, no extra markdown):""",
    input_variables=["context", "question"]
)
    
    # Choose retriever based on hybrid flag
    if hybrid:
        from hybrid_retriever import get_cached_documents
        all_docs = get_cached_documents(vectorstore)   # loads only once
        retriever = HybridRetriever(
            vectorstore=vectorstore,
            documents=all_docs,
            k=k,
            bm25_weight=bm25_weight,
            index_dir="./whoosh_cache"
        )
    else:
        print("Using standard semantic retriever")
        retriever = UniqueDocsRetriever(vectorstore=vectorstore, k=k)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain, prompt