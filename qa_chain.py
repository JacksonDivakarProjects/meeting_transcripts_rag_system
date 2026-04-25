# qa_chain.py
import os
from typing import List

from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from config import GROQ_API_KEY, GROQ_MODEL, WHOOSH_DIR
from hybrid_retriever import HybridRetriever, get_cached_documents


class UniqueDocsRetriever(BaseRetriever):
    """Simple semantic-only retriever that deduplicates results and adds metadata prefixes."""

    vectorstore: object
    k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=self.k * 3)
        seen: set = set()
        unique: List[Document] = []
        for doc, _ in docs_with_scores:
            if doc.page_content in seen:
                continue
            seen.add(doc.page_content)
            meta = doc.metadata
            speaker = meta.get("speaker", "unknown")
            timestamp = meta.get("timestamp_str", "00:00")
            source = meta.get("source_file", "unknown")
            prefix = f"[Speaker: {speaker}, Time: {timestamp}, File: {source}] "
            unique.append(Document(page_content=prefix + doc.page_content, metadata=meta))
            if len(unique) >= self.k:
                break
        return unique

    async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return self._get_relevant_documents(query)


def get_qa_chain(vectorstore, k: int = 5, hybrid: bool = False, bm25_weight: float = 0.3):
    """
    Build and return a RetrievalQA chain.

    Args:
        vectorstore: Chroma vector store.
        k:           Number of documents to retrieve.
        hybrid:      Use HybridRetriever (BM25 + semantic) when True.
        bm25_weight: Fraction of score from keyword search (0–1).
    """
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.0, max_retries=2, timeout=60)

    prompt = PromptTemplate(
        template="""You are a meeting analyst. Answer the user's question using ONLY the provided context.

Each piece of context begins with a citation number in brackets, e.g., [1], [2].
You MUST reference those numbers when citing a source.
Example: "According to [1], the budget was increased."

If the context does not contain the answer, say: "I cannot find that information in the provided meeting transcripts."

Context:
{context}

Question: {question}

Answer (use citation numbers in brackets; plain text, no extra markdown):""",
        input_variables=["context", "question"],
    )

    if hybrid:
        print("Using HybridRetriever (BM25 + semantic)")
        all_docs = get_cached_documents(vectorstore)
        retriever = HybridRetriever(
            vectorstore=vectorstore,
            documents=all_docs,
            k=k,
            bm25_weight=bm25_weight,
            index_dir=WHOOSH_DIR,
        )
    else:
        print("Using semantic-only retriever")
        retriever = UniqueDocsRetriever(vectorstore=vectorstore, k=k)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain, prompt
