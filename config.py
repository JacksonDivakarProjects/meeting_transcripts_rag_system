import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Chunking parameters
CHUNK_SIZE = 1000       # characters per chunk (~250-300 tokens)
CHUNK_OVERLAP = 200     # overlap between chunks

# Use environment variables or relative paths for Docker
JSON_DIR = os.getenv("JSON_DIR", "./data/json_chunks")
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "./meetingbank_vector_db")
WHOOSH_DIR = os.getenv("WHOOSH_DIR", "./whoosh_cache")
