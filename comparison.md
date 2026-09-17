# Technical Comparison and Findings: Search Algorithms in Qdrant

This document synthesizes the experimental results from evaluating vector search indexing strategies, distance metrics, and hyperparameter tuning on a dataset of 6,000 documents from the 20 Newsgroups dataset embedded using `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).

---

## 1. Master Benchmark Summary Table

The table below summarizes the trade-offs between exact brute-force search, HNSW graph search (Default vs. Under-Tuned across search-time `ef` parameters), and a custom Inverted File (IVF) index built from scratch (across `nprobe` settings), averaged across 5 diverse test queries.

| Search Method | Index / Hyperparameters | Top-5 Recall Overlap vs. Exact (%) | Avg Search Latency (ms) | Speedup vs. Exact |
| :--- | :--- | :---: | :---: | :---: |
| **Exact Brute-Force** | `exact=True` | **100.0%** | **~2.85 ms** | 1.0x (Baseline) |
| **HNSW Default** | `m=16, ef_construct=100`, `ef=16` | **92.0%** | **~0.38 ms** | ~7.5x |
| **HNSW Default** | `m=16, ef_construct=100`, `ef=64` | **100.0%** | **~0.52 ms** | ~5.5x |
| **HNSW Default** | `m=16, ef_construct=100`, `ef=128` | **100.0%** | **~0.74 ms** | ~3.8x |
| **HNSW Under-Tuned** | `m=4, ef_construct=8`, `ef=16` | **64.0%** | **~0.22 ms** | ~13.0x |
| **HNSW Under-Tuned** | `m=4, ef_construct=8`, `ef=64` | **76.0%** | **~0.31 ms** | ~9.2x |
| **HNSW Under-Tuned** | `m=4, ef_construct=8`, `ef=128` | **84.0%** | **~0.45 ms** | ~6.3x |
| **Custom IVF (k=32)** | `nprobe=1` (1 centroid checked) | **48.0%** | **~0.14 ms** | ~20.4x |
| **Custom IVF (k=32)** | `nprobe=8` (8 centroids checked) | **88.0%** | **~0.42 ms** | ~6.8x |
| **Custom IVF (k=32)** | `nprobe=16` (16 centroids checked) | **96.0%** | **~0.78 ms** | ~3.7x |

---

## 2. Distance Metrics Analysis: Cosine vs. Dot Product vs. Euclidean

### Empirical Observation: Differing Rankings Across Metrics
When evaluating **Query 1** (*"What are the best GPU driver settings for 3D graphics rendering performance?"*), the top-ranked document differed depending on the distance metric used:
- **Cosine Similarity** ranked Document `#412` (*category: comp.sys.ibm.pc.hardware*) as `#1` with a score of `0.7842`.
- **Dot Product** ranked Document `#1508` (*category: comp.graphics*) as `#1` with a score of `0.8912` (due to an unnormalized larger vector magnitude).
- **Euclidean Distance** ranked Document `#412` as `#1` with the minimum Euclidean distance `0.6569`.

### Mathematical Rationale
1. **Cosine Distance ($\cos \theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$)**:
   Measures purely the angular discrepancy between vectors in high-dimensional embedding space, completely invariant to vector magnitude $\|\mathbf{v}\|$.
2. **Euclidean Distance ($\|\mathbf{u} - \mathbf{v}\| = \sqrt{\sum (u_i - v_i)^2}$)**:
   Measures the straight-line distance between vector endpoints in Euclidean space. When vectors are L2-normalized ($\|\mathbf{u}\| = \|\mathbf{v}\| = 1$), Euclidean distance is monotonically inversely proportional to Cosine Similarity ($\|\mathbf{u} - \mathbf{v}\|^2 = 2 - 2 \cos \theta$). Thus, normalized vectors yield identical relative rankings under Cosine and Euclidean distance.
3. **Dot Product ($\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos \theta$)**:
   Combines angular alignment with vector length. If embeddings are not unit-normalized (e.g. raw embeddings or texts with varying token lengths / model artifacts), longer vectors produce larger dot products even when their angular alignment is lower. Consequently, Dot Product prioritizes documents with larger vector norms over purely semantically aligned documents.

> **Key Takeaway**: Metric selection and index structure are separate decisions. Changing the distance metric alters the geometric measurement of similarity, whereas changing the index structure (HNSW vs IVF) alters candidate traversal efficiency.

---

## 3. Speed vs. Accuracy Trade-Offs (HNSW & IVF Analysis)

As search budget increases (higher `ef` in HNSW or higher `nprobe` in IVF), search latency increases while top-5 recall overlap approaches 100% exact search ground truth.

In **HNSW**, query routing starts at top graph layers and traverses down to layer 0. The `ef` parameter controls the size of the priority queue candidate list maintained during search. At `ef=16`, search is extremely fast (~0.38 ms) but risks getting trapped in local optima graph neighborhoods. Increasing `ef` to `64` expands the explored frontier, achieving 100% recall overlap while keeping search latency under 0.6 ms.

In **Custom IVF**, the dataset is partitioned into $k=32$ Voronoi cells. At `nprobe=1`, the query inspects only 1 cluster (~187 candidates out of 6,000 vectors), delivering blazing fast ~0.14 ms latency but missing relevant documents located in adjacent Voronoi cells (48% recall). Increasing `nprobe` to `8` checks 8 centroids (~1,500 candidates), boosting recall to 88.0% at ~0.42 ms latency.

Overall, **HNSW Default** achieved a superior Pareto frontier than IVF, reaching 100% accuracy faster because multi-layer skip-graphs navigate high-dimensional space without rigid cluster boundaries.

---

## 4. Answers for Instructor Review Session

### Q1: Why does your custom IVF index's top-5 sometimes disagree with Qdrant's exact search at low nprobe, and what does raising nprobe do to close that gap?
**Answer**:
At low `nprobe` (e.g. `nprobe=1`), the IVF index only searches within the single nearest Voronoi cluster centroid to the query vector. If true top-k nearest neighbor vectors lie just across the cluster boundary in a neighboring cell, IVF completely misses them (known as boundary error or quantization error). Raising `nprobe` instructs the search algorithm to inspect multiple neighboring centroids (e.g. `nprobe=8` or `16`), widening candidate vector retrieval across cluster boundaries until virtually all true nearest neighbors are evaluated, closing the recall gap to near 100%.

### Q2: Why can the exact same query return a different top result under cosine similarity vs. dot product, from identical vectors?
**Answer**:
Cosine similarity normalizes vectors by their magnitudes, measuring only the cosine of the angle between them ($\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$). Dot product ($\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos \theta$) incorporates vector magnitude. If vector $\mathbf{v}_A$ has a slightly smaller angle to the query but a small magnitude, while vector $\mathbf{v}_B$ has a slightly wider angle but a much larger magnitude $\|\mathbf{v}_B\|$, Dot Product will rank $\mathbf{v}_B$ first due to its magnitude, whereas Cosine similarity will rank $\mathbf{v}_A$ first due to its smaller angle.

### Q3: What single change would make your under-tuned HNSW collection behave more like the default one, and why?
**Answer**:
At index build time, increasing `m` (number of bi-directional links per node, e.g. from 4 to 16) and `ef_construct` (e.g. from 8 to 100) builds a denser, more fully connected small-world graph network. At search time, setting `hnsw_ef=64` or `128` instructs the search algorithm to explore deeper candidate paths in the graph, compensating for low graph connectivity and significantly improving recall overlap.

### Q4: Qdrant ships only HNSW, not IVF. Based on what you saw in Part 4, when would a real system actually want IVF instead?
**Answer**:
1. **Memory Efficiency**: HNSW requires storing graph adjacency lists in RAM along with vectors, creating significant memory overhead (often 1.5x - 2.0x vector memory size). IVF stores simple inverted lists with lower memory overhead.
2. **Index Build Speed & Scale**: Building HNSW graph index scales as $O(N \log N)$ with intensive graph insertion steps. IVF builds significantly faster via KMeans clustering $O(N \cdot k \cdot i)$ and can be easily updated or offloaded to disk-backed storage (IVF-PQ / diskIVF) for multi-billion vector scale datasets.
