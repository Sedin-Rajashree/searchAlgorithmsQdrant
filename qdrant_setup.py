import os
import json
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, HnswConfigDiff

DATA_DIR = "data"
DOCS_PATH = os.path.join(DATA_DIR, "documents.json")
VECTORS_PATH = os.path.join(DATA_DIR, "vectors.npy")
QUERY_VECTORS_PATH = os.path.join(DATA_DIR, "query_vectors.npy")
QDRANT_URL = "http://localhost:6333"
LOCAL_QDRANT_PATH = "./qdrant_storage_db"

def get_qdrant_client():
    """Attempt connection to Docker server at http://localhost:6333; fallback to local storage if unavailable."""
    try:
        client = QdrantClient(url=QDRANT_URL, timeout=5.0)
        client.get_collections()
        print(f"Connected to Qdrant server at {QDRANT_URL}")
        return client
    except Exception as e:
        print(f"Could not connect to Qdrant at {QDRANT_URL} ({e}). Using embedded local storage: {LOCAL_QDRANT_PATH}")
        return QdrantClient(path=LOCAL_QDRANT_PATH)

def setup_qdrant_collections():
    if not os.path.exists(DOCS_PATH) or not os.path.exists(VECTORS_PATH):
        raise FileNotFoundError("Missing data files. Run data.py and embed.py first.")

    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)
    vectors = np.load(VECTORS_PATH)

    num_vectors, dim = vectors.shape
    print(f"Preparing to index {num_vectors} vectors of dimension {dim}...")

    client = get_qdrant_client()

    collections = {
        "newsgroups_cosine": VectorParams(size=dim, distance=Distance.COSINE),
        "newsgroups_euclidean": VectorParams(size=dim, distance=Distance.EUCLID),
        "newsgroups_dot": VectorParams(size=dim, distance=Distance.DOT),
        "newsgroups_hnsw_default": VectorParams(size=dim, distance=Distance.COSINE),
        "newsgroups_hnsw_undertuned": VectorParams(size=dim, distance=Distance.COSINE),
    }

    hnsw_configs = {
        "newsgroups_hnsw_undertuned": HnswConfigDiff(m=4, ef_construct=8)
    }

    # Batch upsert setup
    batch_size = 500
    points = [
        PointStruct(
            id=doc["id"],
            vector=vectors[i].tolist(),
            payload={"category": doc["category"], "text": doc["text"], "doc_id": doc["id"]}
        )
        for i, doc in enumerate(documents)
    ]

    for coll_name, vector_params in collections.items():
        if client.collection_exists(coll_name):
            print(f"Collection '{coll_name}' already exists. Re-creating...")
            client.delete_collection(coll_name)

        hnsw_cfg = hnsw_configs.get(coll_name, None)
        client.create_collection(
            collection_name=coll_name,
            vectors_config=vector_params,
            hnsw_config=hnsw_cfg
        )
        print(f"Created collection '{coll_name}'")

        # Upsert points
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(collection_name=coll_name, points=batch)

        print(f"Upserted {len(points)} points into '{coll_name}'.")

    print("All Qdrant collections successfully setup and populated!")
    return client

if __name__ == "__main__":
    setup_qdrant_collections()
