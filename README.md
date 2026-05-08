# Hybrid Search Engine

A full-stack search system demonstrating **keyword**, **semantic**, and **hybrid retrieval** — built to compare how each method surfaces results across the same corpus.

Combines an inverted-index TF-IDF ranker with semantic vector search using sentence embeddings and FAISS, with results merged through Reciprocal Rank Fusion (RRF). Deployed with a FastAPI backend and a React (Vite) frontend.
---

## How It Works

| Mode | Method | Strengths |
|---|---|---|
| `keyword` | Inverted index + smoothed IDF scoring | Exact matches, low latency |
| `vector` | `all-MiniLM-L6-v2` embeddings + FAISS inner product | Semantic similarity, handles synonyms |
| `hybrid` | RRF merge of keyword + vector rankings | Best overall relevance |

---

## Stack

**Backend**
- **Python / FastAPI** — `POST /search`, `GET /health`
- **Inverted index** — posting lists with smoothed IDF and cosine-style query scoring (`app/search/inverted_index.py`)
- **Vector store** — [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) + [FAISS](https://github.com/facebookresearch/faiss) on L2-normalized vectors (`app/search/vector_store.py`)
- **Hybrid fusion** — Reciprocal Rank Fusion over keyword and vector result lists (`app/search/hybrid.py`)

**Frontend**
- **React + Vite** — mode switcher, live results, score display
- Dev server proxies `/api/*` to FastAPI on port `8000`

**Corpus**
- `data/corpus.jsonl` — newline-delimited JSON with `id`, `title`, `text` fields

---

## Getting Started

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

> **Note:** The first `vector` or `hybrid` query downloads the embedding model (~90 MB) and builds the FAISS index in memory. Subsequent queries are fast.

To use a custom corpus:
```bash
export HYBRID_CORPUS_PATH=/absolute/path/to/corpus.jsonl
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the URL Vite prints — usually `http://127.0.0.1:5173`.

---

## API Reference

### `POST /search`

```json
{
  "query": "GPU matrix inference",
  "mode": "hybrid",
  "top_k": 10
}
```

**`mode`** — `keyword` | `vector` | `hybrid`

**Response** — ranked list of documents with `id`, `title`, `score`, and `retrieval_mode`.

### `GET /health`

Returns `200 OK` when the API is running.

---

## Roadmap

- [ ] Swap hand-rolled TF-IDF for **BM25** (`rank-bm25`) for a stronger keyword baseline
- [ ] Expand corpus to a licensed dataset (Wikipedia slice, arXiv abstracts) with chunked documents
- [ ] Add **nDCG@k** evaluation on a hand-labelled query set
- [ ] Persist FAISS index and embedding cache to disk for faster cold starts
- [ ] Expose RRF `k` constant and result weights as tunable query parameters
