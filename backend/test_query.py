from sentence_transformers import SentenceTransformer
import faiss, json, numpy as np

print("Loading model and index...")
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index('backend/embeddings/paper_index.faiss')
id_map = json.load(open('backend/embeddings/id_map.json'))

def search(query, top_k=10):
    qvec = model.encode([query]).astype('float32')
    distances, indices = index.search(qvec, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        paper = id_map[idx]
        results.append({**paper, "score": float(dist)})
    return results

queries = [
    "foundational models for transfer learning",
    "training neural networks with limited labeled data",
    "explainability in black box machine learning models"
]

for query in queries:
    print(f"\n=== Query: {query} ===\n")
    for r in search(query):
        print(r['title'], '-', round(r['score'], 3))
