from dotenv import load_dotenv
 
import os
load_dotenv()



GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")  # Default to a powerful model if not set
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")  # Default to a reliable embedding model


JSON_DIR = r"C:\Users\JacksonDivakarRajasi\Documents\Python Folder\json_splitter\datasplit_topics"
VECTOR_DB_DIR = "./meetingbank_vector_db"

# Chunking parameters
CHUNK_SIZE = 1000          # characters per chunk (~250-300 tokens)
CHUNK_OVERLAP = 200        # overlap between chunks
