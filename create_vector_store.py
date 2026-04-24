# process_all.py
from vector_store import build_vector_store

if __name__ == "__main__":
    print("Building vector store from all JSON files...")
    vectorstore = build_vector_store()
    print("Done. You can now run query.py to ask questions.")