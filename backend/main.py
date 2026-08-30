from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import faiss
import json

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
print("Ready.")

@app.get("/search")
def search(q: str, top_k: int = 10):
    qvec = model.encode([q]).astype('float32')
    distances, indices = index.search(qvec, top_k)
    results = [
        {**id_map[idx], "score": float(dist)}
        for dist, idx in zip(distances[0], indices[0])
    ]
    return results

