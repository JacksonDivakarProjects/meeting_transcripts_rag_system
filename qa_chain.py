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


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def normalize_speaker(speaker):
    if not speaker:
        return "Unknown"
    speaker = str(speaker)
    if speaker.lower().startswith("speaker_"):
        return speaker.replace("speaker_", "Speaker ")
    return speaker


def normalize_timestamp(ts):
    if not ts:
        return "00:00"
    try:
        ts = str(ts)
        if ":" in ts:
            return ts
        sec = int(float(ts))
        m, s = divmod(sec, 60)
        return f"{m:02d}:{s:02d}"
    except:
        return "00:00"


# ---------------------------------------------------------
# Retriever (FIXED)
# ---------------------------------------------------------
class UniqueDocsRetriever(BaseRetriever):
    vectorstore: object
    k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        docs_with_scores = self.vectorstore.similarity_search_with_score(
            query, k=self.k * 3
        )

        seen = set()
        unique_docs: List[Document] = []

        for doc, _ in docs_with_scores:
            content = doc.page_content.strip()

            if content in seen:
                continue
            seen.add(content)

            meta = doc.metadata or {}

            # ---- SAFE METADATA ----
            speaker = normalize_speaker(meta.get("speaker"))
            timestamp = normalize_timestamp(meta.get("timestamp_str"))
            source = meta.get("source_file", "unknown")

            prefix = f"[Speaker: {speaker}, Time: {timestamp}, File: {source}] "

            unique_docs.append(
                Document(
                    page_content=prefix + content,
                    metadata=meta
                )
            )

            if len(unique_docs) >= self.k:
                break

        return unique_docs

    async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return self._get_relevant_documents(query)


# ---------------------------------------------------------
# QA Chain
# ---------------------------------------------------------
def get_qa_chain(vectorstore, k: int = 5, hybrid: bool = False, bm25_weight: float = 0.3):

    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.0,
        max_retries=2,
        timeout=60,
    )

    prompt = PromptTemplate(
        template="""You are a meeting analyst. Answer the user's question using ONLY the provided context.

Each piece of context begins with a citation number in brackets, e.g., [1], [2].
You MUST reference those numbers when citing a source.
Example: "According to [1], the budget was increased."

If the context does not contain the answer, say:
"I cannot find that information in the provided meeting transcripts."

Context:
{context}

Question: {question}

Answer (use citation numbers in brackets; plain text only):""",
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