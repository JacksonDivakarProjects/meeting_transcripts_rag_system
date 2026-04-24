import json
import os
import glob
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from config import JSON_DIR, VECTOR_DB_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

def load_and_chunk_one_file(file_path: str) -> List[Document]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    lines = []
    seen = set()
    for topic_name, topic_content in data.items():
        for line in topic_content.get("transcripts", []):
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
                "speaker": line.get("speaker", "unknown"),
                "timestamp_str": timestamp_str,
                "line_id": line_id,
            }
            lines.append(Document(page_content=text, metadata=metadata))
    
    if not lines:
        return []
    
    full_text = "\n".join([doc.page_content for doc in lines])
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    chunk_texts = splitter.split_text(full_text)
    
    chunk_docs = []
    for chunk in chunk_texts:
        first_line = lines[0]
        for line in lines:
            if chunk.startswith(line.page_content[:50]):
                first_line = line
                break
        meta = first_line.metadata.copy()
        meta["chunk_size"] = len(chunk)
        chunk_docs.append(Document(page_content=chunk, metadata=meta))
    
    return chunk_docs

def build_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings
    )
    
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

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings
    )
    print(f"Loaded vector store with {vectorstore._collection.count()} chunks.")
    return vectorstore

def load_all_chunks(batch_size=500) -> List[Document]:
    """Load all document chunks from vector store """
    vectorstore = load_vector_store()
    all_docs = []
    offset = 0
    
    try:
        while True:
            # Get batch of documents
            result = vectorstore._collection.get(
                include=["documents", "metadatas"],
                limit=batch_size,
                offset=offset
            )
            
            if not result["documents"]:
                break
                
            for content, metadata in zip(result["documents"], result["metadatas"]):
                doc = Document(page_content=content, metadata=metadata)
                all_docs.append(doc)
            
            offset += batch_size
            print(f"  Loaded {len(all_docs)} chunks so far...")
            
    except Exception as e:
        print(f"Error loading chunks: {e}")
    
    print(f" Loaded {len(all_docs)} chunks for BM25 index")
    return all_docs