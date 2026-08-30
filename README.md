# ScholarSearch

A semantic search engine for ML research papers. Instead of matching keywords, it understands the *meaning* of your query and finds conceptually related papers — even if they don't share exact words.

## Stack

- **Data**: ~1000 papers fetched from arXiv (`arxiv` Python package)
- **Embeddings**: `all-MiniLM-L6-v2` (sentence-transformers) — converts paper title+abstract into 384-dim vectors
- **Search**: FAISS (`IndexFlatL2`) for fast nearest-neighbor lookup
- **Backend**: FastAPI serving a `/search` endpoint
- **Frontend**: Plain HTML/JS

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r backend/requirements.txt
```

## Running it

**Backend** (from project root):
```bash
uvicorn backend.main:app --reload --port 8001
```

**Frontend** (in a separate terminal, from `/frontend`):
```bash
cd frontend
python -m http.server 5500
```

Then open `http://localhost:5500` in your browser. Make sure the backend is running first.

## Example queries showing semantic search in action

- **"training neural networks with limited labeled data"** → surfaces papers on self-supervised learning, label noise robustness, and semi-supervised continual learning — none using the literal phrase "limited labeled data."
- **"explainability in black box machine learning models"** → surfaces papers titled "Explainable Deep Learning," "Deep Visual Explanation," and "Foundations of Interpretable Models" — related concepts, different wording.
- Note: papers containing your *exact* query words don't always rank first. That's expected — the system ranks by conceptual closeness across the whole abstract, not by literal keyword presence.

## Rebuilding the index

If you want to regenerate everything from scratch:
```bash
python backend/fetch_papers.py     # re-fetch papers from arXiv
python backend/build_index.py      # re-embed + rebuild FAISS index
```

