# hybrid_retriever.py
import os
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, ID, TEXT
from whoosh.qparser import QueryParser

# Global cache for documents, loaded only once
_cached_documents = None

def get_cached_documents(vectorstore):
    global _cached_documents
    if _cached_documents is None:
        from vector_store import load_all_chunks
        print("📚 Loading all chunks (cached once)...")
        _cached_documents = load_all_chunks()
    return _cached_documents


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining Whoosh (keyword search) and semantic vector search.
    Index is cached to disk after first build.
    """
    vectorstore: any = Field(description="Vector store for semantic search")
    documents: List[Document] = Field(default_factory=list, description="All document chunks")
    k: int = Field(default=5, description="Number of documents to retrieve")
    bm25_weight: float = Field(default=0.3, description="Weight for keyword search")
    index_dir: str = Field(default="./whoosh_cache", description="Cache directory for Whoosh index")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, vectorstore, documents: List[Document], k: int = 5, bm25_weight: float = 0.3, index_dir: str = "./whoosh_cache", **kwargs):
        # Check if a valid Whoosh index already exists
        index_exists = False
        if os.path.exists(index_dir) and any(f.endswith(('.seg', '.toc')) for f in os.listdir(index_dir)):
            try:
                # Try to open the index; if successful, it's valid
                open_dir(index_dir)
                index_exists = True
                print(f"📚 Loading existing Whoosh index from {index_dir}...")
            except Exception as e:
                print(f"⚠️ Whoosh index at {index_dir} is corrupted, will rebuild. Error: {e}")
                index_exists = False

        if not index_exists:
            print(f"📚 Building Whoosh index with {len(documents)} documents...")
            os.makedirs(index_dir, exist_ok=True)
            schema = Schema(doc_id=ID(stored=True), content=TEXT(stored=True))
            index = create_in(index_dir, schema)
            writer = index.writer()
            batch_size = 5000
            total = len(documents)
            for i, doc in enumerate(documents):
                writer.add_document(doc_id=str(i), content=doc.page_content)
                if (i + 1) % batch_size == 0:
                    writer.commit()
                    writer = index.writer()
                    print(f"  Indexed {i+1}/{total} documents...")
            writer.commit()
            print(f"✅ Whoosh index built and cached to {index_dir}")
        else:
            index = open_dir(index_dir)

        # Initialize base class with declared fields
        super().__init__(
            vectorstore=vectorstore,
            documents=documents,
            k=k,
            bm25_weight=bm25_weight,
            index_dir=index_dir,
            **kwargs
        )

        # Store index in a private attribute (bypass Pydantic)
        object.__setattr__(self, '_index', index)

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        # 1. Whoosh keyword search
        keyword_results = []
        with self._index.searcher() as searcher:
            parser = QueryParser("content", self._index.schema)
            q = parser.parse(query)
            results = searcher.search(q, limit=self.k * 3)
            for rank, hit in enumerate(results):
                doc_id = int(hit["doc_id"])
                if doc_id < len(self.documents):
                    keyword_results.append((doc_id, 1.0 / (rank + 1)))

        # 2. Vector search (semantic)
        vector_results = self.vectorstore.similarity_search_with_score(query, k=self.k * 3)

        # 3. Reciprocal Rank Fusion
        combined_scores = {}
        for doc_id, rank_score in keyword_results:
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + self.bm25_weight * rank_score
        for rank, (doc, _) in enumerate(vector_results):
            doc_idx = next((i for i, d in enumerate(self.documents) if d.page_content == doc.page_content), None)
            if doc_idx is not None:
                combined_scores[doc_idx] = combined_scores.get(doc_idx, 0) + (1 - self.bm25_weight) * (1.0 / (rank + 1))

        # 4. Get top-k unique, add numbered citation prefixes
        sorted_indices = sorted(combined_scores.keys(), key=lambda i: combined_scores[i], reverse=True)[:self.k]
        unique_docs = []
        seen = set()
        for idx in sorted_indices:
            doc = self.documents[idx]
            if doc.page_content in seen:
                continue
            seen.add(doc.page_content)
            prefix = f"[{len(unique_docs) + 1}] "
            new_doc = Document(
                page_content=prefix + doc.page_content,
                metadata=doc.metadata
            )
            unique_docs.append(new_doc)
        return unique_docs

    async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return self._get_relevant_documents(query)