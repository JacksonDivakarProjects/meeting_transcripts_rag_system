# create_vector_store.py
# Run this once to build the Chroma vector store from your JSON transcript files.
# Usage:  python create_vector_store.py

from vector_store import build_vector_store

if __name__ == "__main__":
    print("Building vector store from all JSON files in JSON_DIR...")
    vectorstore = build_vector_store()
    print("Done! You can now start the API with:  uvicorn main:app --reload")
