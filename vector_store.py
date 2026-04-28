import json
import os
import glob
import re
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from config import JSON_DIR, VECTOR_DB_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


# ---------------------------------------------------------
# Embeddings
# ---------------------------------------------------------
def _get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ---------------------------------------------------------
# Speaker inference (fallback)
# ---------------------------------------------------------
def infer_speaker(text: str) -> str:
    """
    Try to extract speaker from patterns like:
    'John: ...', 'MAYOR: ...'
    """
    match = re.match(r"^([A-Z][A-Za-z .'-]{1,40}):", text)
    if match:
        return match.group(1).strip()
    return "Unknown"


# ---------------------------------------------------------
# Timestamp extraction (robust)
# ---------------------------------------------------------
def get_timestamp(line: dict) -> str:
    if "start_s" in line:
        sec = int(line.get("start_s", 0))
        m, s = divmod(sec, 60)
        return f"{m:02d}:{s:02d}"

    if "start_time" in line:
        return str(line["start_time"])

    if "timestamp" in line:
        return str(line["timestamp"])

    return "00:00"


# ---------------------------------------------------------
# Core parsing
# ---------------------------------------------------------
def _parse_transcript_lines(transcripts: list, topic_name: str, meeting_id: str, file_path: str) -> List[Document]:
    docs = []
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

        # Speaker extraction
        speaker = line.get("speaker")
        if not speaker or speaker.lower() == "unknown":
            speaker = infer_speaker(text)

        # Timestamp extraction
        timestamp_str = get_timestamp(line)

        metadata = {
            "source_file": os.path.basename(file_path),
            "topic": topic_name,
            "meeting_id": meeting_id,
            "speaker": speaker,
            "timestamp_str": timestamp_str,
            "line_id": line_id,
        }

        docs.append(Document(page_content=text, metadata=metadata))

    return docs


# ---------------------------------------------------------
# Chunking
# ---------------------------------------------------------
def load_and_chunk_one_file(file_path: str) -> List[Document]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_lines: List[Document] = []

    if "transcripts" in data:
        topic_name = data.get("topic_name", "Unknown Topic")
        meeting_id = data.get("meeting_id", "")
        all_lines = _parse_transcript_lines(data["transcripts"], topic_name, meeting_id, file_path)
    else:
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


# ---------------------------------------------------------
# Build DB
# ---------------------------------------------------------
def build_vector_store() -> Chroma:
    embeddings = _get_embeddings()

    if os.path.exists(VECTOR_DB_DIR):
        import shutil
        shutil.rmtree(VECTOR_DB_DIR)

    vectorstore = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings
    )

    json_files = glob.glob(os.path.join(JSON_DIR, "*.json"))

    print(f"Found {len(json_files)} files.")

    for i, file_path in enumerate(json_files, 1):
        print(f"Processing {i}: {os.path.basename(file_path)}")

        chunks = load_and_chunk_one_file(file_path)

        if chunks:
            vectorstore.add_documents(chunks)

    print(f"\nFinal chunk count: {vectorstore._collection.count()}")
    return vectorstore


# ---------------------------------------------------------
# Load DB
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
# Load all chunks
# ---------------------------------------------------------
def load_all_chunks(batch_size: int = 500) -> List[Document]:
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