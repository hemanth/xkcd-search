# XKCD Reverse Lookup 
> Upload an XKCD comic image or describe it in text — instantly find which comic it is.

Powered by **TypeSafe AI System One (Jev)** for ultra-fast semantic search with calibrated probabilities, **EmbeddingGemma 2 (`google/embeddinggemma-2`)** & **gemini-embedding-2-preview** multimodal embeddings, and the [olivierdehaene/xkcd](https://huggingface.co/datasets/olivierdehaene/xkcd) dataset.

![Architecture](arch.png?v=3)

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure keys and/or embedding provider
cp .env.example .env
# edit .env:
# TYPESAFE_API_KEY=your_typesafe_key
# GEMINI_API_KEY=your_gemini_key          # Optional if using local EmbeddingGemma 2
# EMBED_PROVIDER=gemma                    # "gemma" (local google/embeddinggemma-2) or "gemini" (gemini-embedding-2-preview API)
# EMBED_MODEL=google/embeddinggemma-2

# Fetch comics from HF dataset (default: last 50)
python fetch_xkcd.py 50

# Optional: Pre-build vector index
python index_comics.py

# Optional: Benchmark TypeSafe speed and throughput
python benchmark_speed.py

# Start the server
python app.py
```

Open [http://localhost:8000](http://localhost:8000) and search!

## Search Engines

1. **TypeSafe AI (Jev System One)** — Evaluates candidate comics in a single round-trip (~100–150ms). Returns calibrated probability distributions, confidence levels, and an existence check (`has_match` Noul). Requires zero vector pre-indexing.
2. **EmbeddingGemma 2 / Gemini Vector Search** — Embeds text or uploaded comic images into a unified 768d space:
   - **Local (`EMBED_PROVIDER=gemma` or no `GEMINI_API_KEY`):** Runs `google/embeddinggemma-2` locally in `bfloat16` via `SentenceTransformer` and computes cosine similarity directly over `embeddings/comics.npz` (auto-cached on first run, no ChromaDB required) or ChromaDB if indexed.
   - **Cloud (`EMBED_PROVIDER=gemini` or when `GEMINI_API_KEY` is set):** Uses `gemini-embedding-2-preview` via `google-genai`.
3. **Hybrid Re-ranking** — Uses vector search to retrieve candidate shortlists, then applies TypeSafe System One (Jev) to re-rank the shortlist with calibrated decision probabilities.

## Speed Benchmark

To benchmark TypeSafe AI latency and concurrency:

```bash
python benchmark_speed.py
```

Or test it live right in the browser using the **Benchmark Speed** button in the web UI.

