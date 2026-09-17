import time
import numpy as np
from sklearn.cluster import KMeans

class IVFIndex:
    """
    Inverted File (IVF) vector index built from scratch.
    
    1. Clusters vectors into k Voronoi cells using KMeans.
    2. Maintains inverted lists mapping cluster_id -> vector_ids.
    3. At search time, checks only the 'nprobe' nearest clusters.
    """
    def __init__(self, n_clusters=32, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.centroids = None
        self.inverted_lists = {}
        self.vectors = None
        self.is_built = False

    def build_index(self, vectors: np.ndarray):
        """Train KMeans and populate inverted index."""
        print(f"Building IVF index with k={self.n_clusters} clusters on {len(vectors)} vectors...")
        start_time = time.perf_counter()
        
        self.vectors = vectors.astype(np.float32)
        # Ensure vectors are normalized for cosine distance calculations
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.normalized_vectors = self.vectors / norms
        
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        labels = self.kmeans.fit_predict(self.normalized_vectors)
        self.centroids = self.kmeans.cluster_centers_
        
        # Build inverted lists mapping cluster_id -> list of vector indices
        self.inverted_lists = {c: [] for c in range(self.n_clusters)}
        for doc_id, cluster_id in enumerate(labels):
            self.inverted_lists[cluster_id].append(doc_id)
            
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self.is_built = True
        print(f"IVF index built successfully in {elapsed_ms:.2f} ms.")
        return elapsed_ms

    def search(self, query_vector: np.ndarray, nprobe: int = 1, top_k: int = 5):
        """
        Search the IVF index using nprobe candidate clusters.
        
        Returns:
            top_results: list of tuples (doc_id, score)
            latency_ms: float
            candidates_checked: int
        """
        if not self.is_built:
            raise RuntimeError("IVF index is not built. Call build_index() first.")
            
        start_time = time.perf_counter()
        
        # 1. Normalize query vector
        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm
            
        # 2. Find nprobe nearest centroids (using cosine similarity to centroids)
        centroid_norms = np.linalg.norm(self.centroids, axis=1)
        centroid_norms[centroid_norms == 0] = 1e-10
        centroid_sims = np.dot(self.centroids, q) / centroid_norms
        
        # Top nprobe cluster IDs
        nearest_clusters = np.argsort(centroid_sims)[::-1][:nprobe]
        
        # 3. Gather candidate vector indices
        candidate_ids = []
        for cid in nearest_clusters:
            candidate_ids.extend(self.inverted_lists[cid])
            
        candidates_checked = len(candidate_ids)
        
        if not candidate_ids:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return [], latency_ms, 0
            
        # 4. Exact search over candidate vectors
        candidate_vectors = self.normalized_vectors[candidate_ids]
        scores = np.dot(candidate_vectors, q)
        
        # 5. Get top_k candidates
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = [
            (int(candidate_ids[idx]), float(scores[idx]))
            for idx in top_indices
        ]
        
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return results, latency_ms, candidates_checked

if __name__ == "__main__":
    import os
    VECTORS_PATH = os.path.join("data", "vectors.npy")
    QUERY_VECTORS_PATH = os.path.join("data", "query_vectors.npy")
    
    if os.path.exists(VECTORS_PATH) and os.path.exists(QUERY_VECTORS_PATH):
        vecs = np.load(VECTORS_PATH)
        q_vecs = np.load(QUERY_VECTORS_PATH)
        ivf = IVFIndex(n_clusters=32)
        ivf.build_index(vecs)
        
        for nprobe in [1, 4, 8, 16]:
            results, lat, cands = ivf.search(q_vecs[0], nprobe=nprobe, top_k=5)
            print(f"nprobe={nprobe:2d} | latency: {lat:.3f} ms | candidates: {cands:4d} | top match doc_id: {results[0][0]} score: {results[0][1]:.4f}")
    else:
        print("Vectors not found. Run data.py and embed.py first.")
