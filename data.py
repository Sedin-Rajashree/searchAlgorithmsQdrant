import os
import json
import random
from sklearn.datasets import fetch_20newsgroups

DATA_DIR = "data"
OUTPUT_PATH = os.path.join(DATA_DIR, "documents.json")
SAMPLE_SIZE = 6000
SEED = 42

def prepare_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("Fetching 20 Newsgroups dataset (subset='all')...")
    newsgroups = fetch_20newsgroups(
        subset='all',
        remove=('headers', 'footers', 'quotes'),
        download_if_missing=True
    )
    
    documents = []
    category_names = newsgroups.target_names
    
    for idx, (text, category_idx) in enumerate(zip(newsgroups.data, newsgroups.target)):
        cleaned_text = text.strip()
        # Filter out empty or trivially short documents
        if len(cleaned_text) < 30:
            continue
        
        documents.append({
            "id": len(documents),
            "original_id": idx,
            "category": category_names[category_idx],
            "text": cleaned_text[:1500]  # Truncate extremely long docs to 1500 chars for efficiency
        })
    
    print(f"Total valid documents loaded: {len(documents)}")
    
    # Reproducible sampling
    random.seed(SEED)
    if len(documents) > SAMPLE_SIZE:
        sampled_docs = random.sample(documents, SAMPLE_SIZE)
    else:
        sampled_docs = documents
        
    # Re-index sampled documents 0..N-1
    for i, doc in enumerate(sampled_docs):
        doc["id"] = i
        
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(sampled_docs, f, indent=2)
        
    print(f"Saved {len(sampled_docs)} documents to '{OUTPUT_PATH}'.")
    return sampled_docs

if __name__ == "__main__":
    prepare_dataset()
