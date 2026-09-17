import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
DOCS_PATH = os.path.join(DATA_DIR, "documents.json")
VECTORS_PATH = os.path.join(DATA_DIR, "vectors.npy")
QUERY_VECTORS_PATH = os.path.join(DATA_DIR, "query_vectors.npy")
QUERIES_FILE = "queries.md"

MODEL_NAME = "all-MiniLM-L6-v2"

# 5 Default test queries as specified in queries.md
QUERIES = [
    "What are the best GPU driver settings for 3D graphics rendering performance?",
    "How does public-key cryptography and RSA encryption keep web data secure?",
    "What are the latest discoveries about planetary orbits and NASA space shuttle launches?",
    "What is the standard maintenance procedure for adjusting motorcycle clutch cables?",
    "How do medical researchers diagnose autoimmune diseases and treat viral infections?"
]

def generate_embeddings():
    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(f"'{DOCS_PATH}' not found. Please run data.py first.")
        
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)
        
    texts = [doc["text"] for doc in documents]
    print(f"Loading embedding model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"Embedding {len(texts)} documents...")
    doc_vectors = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    doc_vectors = np.array(doc_vectors, dtype=np.float32)
    
    print(f"Embedding {len(QUERIES)} test queries...")
    query_vectors = model.encode(QUERIES, show_progress_bar=False, normalize_embeddings=True)
    query_vectors = np.array(query_vectors, dtype=np.float32)
    
    np.save(VECTORS_PATH, doc_vectors)
    np.save(QUERY_VECTORS_PATH, query_vectors)
    
    print(f"Saved document vectors shape: {doc_vectors.shape} to '{VECTORS_PATH}'")
    print(f"Saved query vectors shape: {query_vectors.shape} to '{QUERY_VECTORS_PATH}'")
    
    return doc_vectors, query_vectors

if __name__ == "__main__":
    generate_embeddings()
