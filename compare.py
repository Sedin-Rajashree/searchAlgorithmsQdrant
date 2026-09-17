import os
import json
import time
import numpy as np
import pandas as pd
from qdrant_client.models import SearchParams, Distance

from qdrant_setup import get_qdrant_client, DATA_DIR, DOCS_PATH, VECTORS_PATH, QUERY_VECTORS_PATH
from ivf import IVFIndex

RESULTS_DIR = "results"

QUERIES = [
    "What are the best GPU driver settings for 3D graphics rendering performance?",
    "How does public-key cryptography and RSA encryption keep web data secure?",
    "What are the latest discoveries about planetary orbits and NASA space shuttle launches?",
    "What is the standard maintenance procedure for adjusting motorcycle clutch cables?",
    "How do medical researchers diagnose autoimmune diseases and treat viral infections?"
]

def calculate_overlap(retrieved_ids, ground_truth_ids):
    """Calculate the overlap percentage of top-k retrieved IDs vs ground truth IDs."""
    if not ground_truth_ids:
        return 0.0
    intersection = set(retrieved_ids).intersection(set(ground_truth_ids))
    return (len(intersection) / len(ground_truth_ids)) * 100.0

def run_benchmarks():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load Data
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)
    doc_lookup = {doc["id"]: doc for doc in documents}
    
    doc_vectors = np.load(VECTORS_PATH)
    query_vectors = np.load(QUERY_VECTORS_PATH)
    
    client = get_qdrant_client()

    print("=" * 70)
    print(" PART 2: DISTANCE METRICS COMPARISON (Cosine, Euclidean, Dot)")
    print("=" * 70)
    
    distance_results = {}
    metrics = ["cosine", "euclidean", "dot"]
    
    for q_idx, query_text in enumerate(QUERIES):
        q_vec = query_vectors[q_idx].tolist()
        distance_results[f"query_{q_idx+1}"] = {
            "query": query_text,
            "metrics": {}
        }
        
        for metric in metrics:
            coll_name = f"newsgroups_{metric}"
            
            start_t = time.perf_counter()
            res = client.query_points(
                collection_name=coll_name,
                query=q_vec,
                limit=5
            ).points
            latency_ms = (time.perf_counter() - start_t) * 1000.0
            
            hits = [
                {
                    "rank": rank + 1,
                    "doc_id": hit.id,
                    "score": round(float(hit.score), 5),
                    "category": hit.payload.get("category"),
                    "text_snippet": hit.payload.get("text", "")[:120] + "..."
                }
                for rank, hit in enumerate(res)
            ]
            distance_results[f"query_{q_idx+1}"]["metrics"][metric] = {
                "latency_ms": round(latency_ms, 3),
                "top_5": hits
            }

    metrics_out_path = os.path.join(RESULTS_DIR, "distance_metrics.json")
    with open(metrics_out_path, "w", encoding="utf-8") as f:
        json.dump(distance_results, f, indent=2)
    print(f"Saved distance metrics results to '{metrics_out_path}'.")

    # Print representative snippet showing ranking differences
    print("\n[Distance Metric Ranking Sample - Query 1]")
    for m in metrics:
        top1 = distance_results["query_1"]["metrics"][m]["top_5"][0]
        print(f" Metric: {m:9s} | Top Match Doc ID: {top1['doc_id']:4d} | Category: {top1['category']:25s} | Score: {top1['score']:.4f}")


    print("\n" + "=" * 70)
    print(" PART 3: EXACT SEARCH vs. HNSW (Default vs. Under-Tuned)")
    print("=" * 70)
    
    # Compute Exact Search Ground Truth (exact=True)
    exact_ground_truth = {}
    exact_latencies = []
    
    for q_idx, q_vec in enumerate(query_vectors):
        start_t = time.perf_counter()
        res = client.query_points(
            collection_name="newsgroups_hnsw_default",
            query=q_vec.tolist(),
            search_params=SearchParams(exact=True),
            limit=5
        ).points
        lat = (time.perf_counter() - start_t) * 1000.0
        exact_latencies.append(lat)
        exact_ground_truth[q_idx] = [hit.id for hit in res]
        
    avg_exact_latency = float(np.mean(exact_latencies))
    print(f"Exact Search Ground Truth computed (Avg Latency: {avg_exact_latency:.3f} ms).")

    hnsw_results = {
        "exact_brute_force": {
            "avg_latency_ms": round(avg_exact_latency, 3),
            "avg_overlap_percent": 100.0
        },
        "experiments": []
    }

    hnsw_configs = [
        ("HNSW Default (m=16, ef_construct=100)", "newsgroups_hnsw_default"),
        ("HNSW Under-Tuned (m=4, ef_construct=8)", "newsgroups_hnsw_undertuned")
    ]
    ef_values = [16, 64, 128]

    for config_label, coll_name in hnsw_configs:
        for ef in ef_values:
            overlaps = []
            latencies = []
            
            for q_idx, q_vec in enumerate(query_vectors):
                start_t = time.perf_counter()
                res = client.query_points(
                    collection_name=coll_name,
                    query=q_vec.tolist(),
                    search_params=SearchParams(hnsw_ef=ef, exact=False),
                    limit=5
                ).points
                lat = (time.perf_counter() - start_t) * 1000.0
                
                retrieved_ids = [hit.id for hit in res]
                overlap = calculate_overlap(retrieved_ids, exact_ground_truth[q_idx])
                
                overlaps.append(overlap)
                latencies.append(lat)
                
            avg_overlap = float(np.mean(overlaps))
            avg_lat = float(np.mean(latencies))
            
            exp_record = {
                "config": config_label,
                "collection": coll_name,
                "hnsw_ef": ef,
                "avg_overlap_percent": round(avg_overlap, 2),
                "avg_latency_ms": round(avg_lat, 3)
            }
            hnsw_results["experiments"].append(exp_record)
            print(f" {config_label:42s} | ef={ef:3d} | Recall Overlap: {avg_overlap:6.2f}% | Latency: {avg_lat:.3f} ms")

    hnsw_out_path = os.path.join(RESULTS_DIR, "hnsw.json")
    with open(hnsw_out_path, "w", encoding="utf-8") as f:
        json.dump(hnsw_results, f, indent=2)
    print(f"Saved HNSW benchmark results to '{hnsw_out_path}'.")


    print("\n" + "=" * 70)
    print(" PART 4: CUSTOM IVF INDEX BENCHMARK (nprobe tuning)")
    print("=" * 70)
    
    ivf = IVFIndex(n_clusters=32, random_state=42)
    ivf_build_time = ivf.build_index(doc_vectors)

    ivf_results = {
        "build_time_ms": round(ivf_build_time, 2),
        "experiments": []
    }

    nprobe_values = [1, 8, 16]

    for nprobe in nprobe_values:
        overlaps = []
        latencies = []
        candidates_list = []
        
        for q_idx, q_vec in enumerate(query_vectors):
            res_hits, lat, cands = ivf.search(q_vec, nprobe=nprobe, top_k=5)
            retrieved_ids = [hit[0] for hit in res_hits]
            overlap = calculate_overlap(retrieved_ids, exact_ground_truth[q_idx])
            
            overlaps.append(overlap)
            latencies.append(lat)
            candidates_list.append(cands)
            
        avg_overlap = float(np.mean(overlaps))
        avg_lat = float(np.mean(latencies))
        avg_cands = float(np.mean(candidates_list))
        
        exp_record = {
            "nprobe": nprobe,
            "avg_candidates_checked": round(avg_cands, 1),
            "avg_overlap_percent": round(avg_overlap, 2),
            "avg_latency_ms": round(avg_lat, 3)
        }
        ivf_results["experiments"].append(exp_record)
        print(f" Custom IVF (k=32) | nprobe={nprobe:2d} | Candidates: {avg_cands:6.1f} | Recall Overlap: {avg_overlap:6.2f}% | Latency: {avg_lat:.3f} ms")

    ivf_out_path = os.path.join(RESULTS_DIR, "ivf.json")
    with open(ivf_out_path, "w", encoding="utf-8") as f:
        json.dump(ivf_results, f, indent=2)
    print(f"Saved IVF benchmark results to '{ivf_out_path}'.")


    print("\n" + "=" * 70)
    print(" PART 5: MASTER COMPARISON SUMMARY TABLE")
    print("=" * 70)

    table_rows = []
    
    # 1. Exact
    table_rows.append({
        "Method": "Exact Brute-Force",
        "Parameters": "exact=True",
        "Top-5 Overlap (%)": f"{hnsw_results['exact_brute_force']['avg_overlap_percent']:.1f}%",
        "Avg Latency (ms)": f"{hnsw_results['exact_brute_force']['avg_latency_ms']:.3f} ms"
    })
    
    # 2. HNSW Default
    for exp in hnsw_results["experiments"]:
        if "Default" in exp["config"]:
            table_rows.append({
                "Method": "HNSW Default (m=16)",
                "Parameters": f"ef={exp['hnsw_ef']}",
                "Top-5 Overlap (%)": f"{exp['avg_overlap_percent']:.1f}%",
                "Avg Latency (ms)": f"{exp['avg_latency_ms']:.3f} ms"
            })
            
    # 3. HNSW Under-Tuned
    for exp in hnsw_results["experiments"]:
        if "Under-Tuned" in exp["config"]:
            table_rows.append({
                "Method": "HNSW Under-Tuned (m=4)",
                "Parameters": f"ef={exp['hnsw_ef']}",
                "Top-5 Overlap (%)": f"{exp['avg_overlap_percent']:.1f}%",
                "Avg Latency (ms)": f"{exp['avg_latency_ms']:.3f} ms"
            })
            
    # 4. Custom IVF
    for exp in ivf_results["experiments"]:
        table_rows.append({
            "Method": "Custom IVF (k=32)",
            "Parameters": f"nprobe={exp['nprobe']}",
            "Top-5 Overlap (%)": f"{exp['avg_overlap_percent']:.1f}%",
            "Avg Latency (ms)": f"{exp['avg_latency_ms']:.3f} ms"
        })

    summary_df = pd.DataFrame(table_rows)
    print(summary_df.to_string(index=False))

    summary_out_path = os.path.join(RESULTS_DIR, "combined_summary.json")
    with open(summary_out_path, "w", encoding="utf-8") as f:
        json.dump(table_rows, f, indent=2)
        
    print("\nBenchmark suite execution complete!")

if __name__ == "__main__":
    run_benchmarks()
