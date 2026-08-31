from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading model and index...")
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index('backend/embeddings/paper_index.faiss')
id_map = json.load(open('backend/embeddings/id_map.json'))
papers = json.load(open('backend/data/papers.json'))  # for abstracts
print("Ready.")

@app.get("/search")
def search(q: str, top_k: int = 10):
    qvec = model.encode([q]).astype('float32')
    faiss.normalize_L2(qvec)  # normalize query vector

    distances, indices = index.search(qvec, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        paper_meta = id_map[idx]
        abstract = papers[idx]['summary']
        similarity = float(1 - (dist / 2))  # convert L2 (on normalized vecs) to cosine similarity
        results.append({
            **paper_meta,
            "abstract": abstract,
            "similarity": round(similarity * 100, 1)  # as a percentage
        })
    return results

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)

    