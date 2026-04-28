import os
import re
from typing import List, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from whoosh.index import create_in, open_dir, exists_in
from whoosh.fields import Schema, ID, TEXT
from whoosh.qparser import QueryParser

from config import WHOOSH_DIR

# Global document cache — loaded only once per process
_cached_documents: List[Document] = []


def get_cached_documents(vectorstore: Any) -> List[Document]:
    global _cached_documents
    if not _cached_documents:
        from vector_store import load_all_chunks
        print("📚 Loading all chunks into memory (cached once)...")
        _cached_documents = load_all_chunks()
    return _cached_documents


def escape_query(query: str) -> str:
    """Escape Whoosh special characters manually."""
    return re.sub(r'([+\-!(){}\[\]^"~*?:\\/])', r'\\\1', query)


class HybridRetriever(BaseRetriever):
    """Combines Whoosh BM25 keyword search with Chroma semantic search via RRF."""

    vectorstore: Any = Field(description="Chroma vector store for semantic search")
    documents: List[Document] = Field(default_factory=list, description="All document chunks")
    k: int = Field(default=5, description="Final number of documents to return")
    bm25_weight: float = Field(default=0.3, description="Weight for keyword results (0-1)")
    index_dir: str = Field(default=WHOOSH_DIR, description="Whoosh index cache directory")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, vectorstore: Any, documents: List[Document],
                 k: int = 5, bm25_weight: float = 0.3,
                 index_dir: str = WHOOSH_DIR, **kwargs):

        os.makedirs(index_dir, exist_ok=True)
        index = None

        # Try to reuse existing index
        if exists_in(index_dir):
            try:
                index = open_dir(index_dir)
                print(f"✅ Reusing Whoosh index from {index_dir} ({index.doc_count()} docs).")
            except Exception as e:
                print(f"⚠️ Corrupt Whoosh index — rebuilding. ({e})")
                index = None

        if index is None:
            print(f"🔨 Building Whoosh index for {len(documents)} documents...")
            schema = Schema(doc_id=ID(stored=True), content=TEXT(stored=True))
            index = create_in(index_dir, schema)
            writer = index.writer()
            batch_size = 5000

            for i, doc in enumerate(documents):
                writer.add_document(doc_id=str(i), content=doc.page_content)

                if (i + 1) % batch_size == 0:
                    writer.commit()
                    writer = index.writer()
                    print(f"  Indexed {i + 1}/{len(documents)}...")

            writer.commit()
            print(f"✅ Whoosh index built ({len(documents)} docs).")

        super().__init__(
            vectorstore=vectorstore,
            documents=documents,
            k=k,
            bm25_weight=bm25_weight,
            index_dir=index_dir,
            **kwargs,
        )

        object.__setattr__(self, "_index", index)

    # ------------------------------------------------------------------
    # Core retrieval logic
    # ------------------------------------------------------------------
    def _retrieve(self, query: str) -> List[Document]:
        fetch = self.k * 3

        # 1. BM25 keyword search
        keyword_hits: List[tuple] = []

        with self._index.searcher() as searcher:
            parser = QueryParser("content", self._index.schema)

            escaped_query = escape_query(query)

            try:
                q = parser.parse(escaped_query)
            except Exception:
                # fallback to raw query
                q = parser.parse(query)

            results = searcher.search(q, limit=fetch)

            for rank, hit in enumerate(results):
                doc_id = int(hit["doc_id"])
                if doc_id < len(self.documents):
                    keyword_hits.append((doc_id, 1.0 / (rank + 1)))

        # 2. Semantic search
        vector_results = self.vectorstore.similarity_search_with_score(query, k=fetch)

        content_to_idx = {doc.page_content: i for i, doc in enumerate(self.documents)}

        # 3. Reciprocal Rank Fusion
        scores: dict = {}

        for doc_id, rr_score in keyword_hits:
            scores[doc_id] = scores.get(doc_id, 0.0) + self.bm25_weight * rr_score

        for rank, (doc, _) in enumerate(vector_results):
            idx = content_to_idx.get(doc.page_content)
            if idx is not None:
                sem_score = (1.0 - self.bm25_weight) * (1.0 / (rank + 1))
                scores[idx] = scores.get(idx, 0.0) + sem_score

        # 4. Top-k selection
        sorted_indices = sorted(
            scores,
            key=lambda i: (-scores[i], i)
        )

        unique_docs: List[Document] = []
        seen: set = set()

        for idx in sorted_indices:
            if len(unique_docs) >= self.k:
                break

            doc = self.documents[idx]

            if doc.page_content in seen:
                continue

            seen.add(doc.page_content)

            citation = f"[{len(unique_docs) + 1}] "

            unique_docs.append(
                Document(
                    page_content=citation + doc.page_content,
                    metadata=doc.metadata
                )
            )

        return unique_docs

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return self._retrieve(query)

    async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._retrieve, query)