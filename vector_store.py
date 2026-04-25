# vector_store.py
import json
import os
import glob
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from config import JSON_DIR, VECTOR_DB_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


def _get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _parse_transcript_lines(transcripts: list, topic_name: str, meeting_id: str, file_path: str) -> List[Document]:
    """Convert raw transcript lines into deduplicated Document objects."""
    lines = []
    seen = set()
    for line in transcripts:
        text = line.get("contents", "").strip()
        if not text:
            continue
        line_id = line.get("line_id", 0)
        key = (topic_name, line_id, text)
        if key in seen:
            continue
        seen.add(key)

        start_sec = line.get("start_s", 0)
        minutes, seconds = divmod(int(start_sec), 60)
        timestamp_str = f"{minutes:02d}:{seconds:02d}"

        metadata = {
            "source_file": os.path.basename(file_path),
            "topic": topic_name,
            "meeting_id": meeting_id,
            "speaker": line.get("speaker", "unknown"),
            "timestamp_str": timestamp_str,
            "line_id": line_id,
        }
        lines.append(Document(page_content=text, metadata=metadata))
    return lines


def load_and_chunk_one_file(file_path: str) -> List[Document]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_lines: List[Document] = []

    if "transcripts" in data:
        # Flat format: single topic per file
        topic_name = data.get("topic_name", "Unknown Topic")
        meeting_id = data.get("meeting_id", "")
        all_lines = _parse_transcript_lines(data["transcripts"], topic_name, meeting_id, file_path)
    else:
        # Nested format: multiple topics per file
        for topic_name, topic_content in data.items():
            if not isinstance(topic_content, dict):
                continue
            meeting_id = topic_content.get("meeting_id", "")
            transcripts = topic_content.get("transcripts", [])
            all_lines.extend(_parse_transcript_lines(transcripts, topic_name, meeting_id, file_path))

    if not all_lines:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunk_docs = []
    for i, doc in enumerate(all_lines):
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            meta = doc.metadata.copy()
            meta["chunk_index"] = i
            meta["chunk_size"] = len(chunk)
            chunk_docs.append(Document(page_content=chunk, metadata=meta))

    return chunk_docs


def build_vector_store() -> Chroma:
    embeddings = _get_embeddings()
    vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)

    json_files = glob.glob(os.path.join(JSON_DIR, "*.json"))
    total = len(json_files)
    print(f"Found {total} JSON files.")

    for idx, file_path in enumerate(json_files, 1):
        print(f"Processing {idx}/{total}: {os.path.basename(file_path)}")
        chunks = load_and_chunk_one_file(file_path)
        if chunks:
            vectorstore.add_documents(chunks)
        if idx % 100 == 0:
            count = vectorstore._collection.count()
            print(f"  -> {count} total chunks in DB so far.")

    print(f"\nVector store built. Total chunks: {vectorstore._collection.count()}")
    return vectorstore


def load_vector_store() -> Chroma:
    embeddings = _get_embeddings()
    vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
    print(f"Loaded vector store with {vectorstore._collection.count()} chunks.")
    return vectorstore


def load_all_chunks(batch_size: int = 500) -> List[Document]:
    """Load every chunk from the persisted Chroma collection."""
    vectorstore = load_vector_store()
    all_docs: List[Document] = []
    offset = 0

    try:
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
            print(f"  Loaded {len(all_docs)} chunks so far...")
    except Exception as e:
        print(f"Error loading chunks: {e}")

    print(f"Loaded {len(all_docs)} chunks total.")
    return all_docs
