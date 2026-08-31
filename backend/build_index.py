from sentence_transformers import SentenceTransformer
import json
import numpy as np

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Loading papers...")
with open('backend/data/papers.json', 'r') as f:
    papers = json.load(f)

texts = [p['title'] + ". " + p['summary'] for p in papers]

print(f"Encoding {len(texts)} papers...")
vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

print("Saving vectors...")
np.save('backend/embeddings/paper_vectors.npy', vectors)

id_map = [
    {
        "title": p["title"],
        "authors": p["authors"],
        "link": p["link"],
        "published": p["published"]
    }
    for p in papers
]

print("Saving id_map...")
with open('backend/embeddings/id_map.json', 'w') as f:
    json.dump(id_map, f, indent=2)

print("Done.")

import faiss

print("Building FAISS index...")
vectors = np.load('backend/embeddings/paper_vectors.npy')
dim = vectors.shape[1]

vectors = vectors.astype('float32')
faiss.normalize_L2(vectors)
index = faiss.IndexFlatL2(dim)
index.add(vectors)

faiss.write_index(index, 'backend/embeddings/paper_index.faiss')
print(f"Index built with {index.ntotal} vectors of dimension {dim}.")

