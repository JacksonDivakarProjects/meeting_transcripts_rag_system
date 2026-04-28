import json
import os
import glob
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from config import JSON_DIR, VECTOR_DB_DIR, EMBEDDING_MODEL

CHUNK_SIZE = 400
CHUNK_OVERLAP = 75


def _get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ---------------------------------------------------------
# Speaker-based merging (CRITICAL FIX)
# ---------------------------------------------------------
def _merge_speaker_lines(transcripts: list, topic_name: str, meeting_id: str, file_path: str) -> List[Document]:
    merged_docs = []
    current_speaker = None
    buffer = []

    for line in transcripts:
        text = line.get("contents", "").strip()
        if not text:
            continue

        # filter noise
        if len(text) < 10:
            continue

        speaker = line.get("speaker", "unknown")

        if speaker != current_speaker:
            if buffer:
                merged_docs.append(" ".join(buffer))
            buffer = [text]
            current_speaker = speaker
        else:
            buffer.append(text)

    if buffer:
        merged_docs.append(" ".join(buffer))

    documents = []
    seen = set()

    for i, content in enumerate(merged_docs):
        content = content.strip()

        if content in seen:
            continue
        seen.add(content)

        metadata = {
            "source_file": os.path.basename(file_path),
            "topic": topic_name,
            "meeting_id": meeting_id,
            "chunk_id": i,
        }

        documents.append(Document(page_content=content, metadata=metadata))

    return documents


# ---------------------------------------------------------
# Chunking (controlled)
# ---------------------------------------------------------
def load_and_chunk_one_file(file_path: str) -> List[Document]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents: List[Document] = []

    if "transcripts" in data:
        topic_name = data.get("topic_name", "Unknown")
        meeting_id = data.get("meeting_id", "")
        documents = _merge_speaker_lines(data["transcripts"], topic_name, meeting_id, file_path)
    else:
        for topic_name, topic_content in data.items():
            if not isinstance(topic_content, dict):
                continue
            meeting_id = topic_content.get("meeting_id", "")
            transcripts = topic_content.get("transcripts", [])
            documents.extend(
                _merge_speaker_lines(transcripts, topic_name, meeting_id, file_path)
            )

    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunk_docs = []

    for doc in documents:
        chunks = splitter.split_text(doc.page_content)

        for chunk in chunks:
            chunk = chunk.strip()

            # remove noise
            if len(chunk) < 50:
                continue

            meta = doc.metadata.copy()
            meta["chunk_len"] = len(chunk)

            chunk_docs.append(Document(page_content=chunk, metadata=meta))

    return chunk_docs


# ---------------------------------------------------------
# Build vector DB (optimized)
# ---------------------------------------------------------
def build_vector_store() -> Chroma:
    embeddings = _get_embeddings()

    # clear old DB (important to avoid duplicates)
    if os.path.exists(VECTOR_DB_DIR):
        import shutil
        shutil.rmtree(VECTOR_DB_DIR)

    vectorstore = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings
    )

    json_files = glob.glob(os.path.join(JSON_DIR, "*.json"))

    print(f"Found {len(json_files)} files.")

    total_chunks = 0

    for i, file_path in enumerate(json_files, 1):
        print(f"Processing {i}: {os.path.basename(file_path)}")

        chunks = load_and_chunk_one_file(file_path)

        if chunks:
            vectorstore.add_documents(chunks)
            total_chunks += len(chunks)

        if i % 50 == 0:
            print(f"  → {total_chunks} chunks so far")

    print(f"\nFinal chunk count: {total_chunks}")
    return vectorstore


# ---------------------------------------------------------
# Load existing DB
# ---------------------------------------------------------
def load_vector_store() -> Chroma:
    embeddings = _get_embeddings()
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings
    )
    print(f"Loaded vector DB: {vectorstore._collection.count()} chunks")
    return vectorstore


# ---------------------------------------------------------
# Load all chunks (ONCE only)
# ---------------------------------------------------------
def load_all_chunks(batch_size: int = 1000) -> List[Document]:
    vectorstore = load_vector_store()

    all_docs: List[Document] = []
    offset = 0

    while True:
        result = vectorstore._collection.get(
            include=["documents", "metadatas"],
            limit=batch_size,
            offset=offset,
        )

        if not result["documents"]:
            break

        for content, metadata in zip(result["documents"], result["metadatas"]):
            all_docs.append(Document(page_content=content, metadata=metadata or {}))

        offset += batch_size
        print(f"Loaded {len(all_docs)} chunks...")

    print(f"Total loaded: {len(all_docs)}")
    return all_docs