import io
import json
import os
import time
from typing import Any

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul

load_dotenv()


_client = None
_chroma = None
_st_model = None

MODEL = os.getenv("EMBED_MODEL", "google/embeddinggemma-2")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-2-preview")
CHROMA_DIR = "chroma_db"
IMAGE_COLLECTION = "xkcd_images"
TEXT_COLLECTION = "xkcd_text"


def _get_client(api_key: str | None = None) -> genai.Client:
    if api_key:
        return genai.Client(api_key=api_key)
    global _client
    if _client is None:
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not set")
        _client = genai.Client(api_key=key)
    return _client


def _get_embeddinggemma_model():
    global _st_model
    if _st_model is None:
        import torch
        from sentence_transformers import SentenceTransformer

        _st_model = SentenceTransformer(MODEL, model_kwargs={"torch_dtype": torch.bfloat16})
    return _st_model


def _use_local_embeddinggemma(api_key: str | None = None) -> bool:
    provider = os.getenv("EMBED_PROVIDER", "").strip().lower()
    if provider in ("gemma", "embeddinggemma", "local"):
        return True
    if provider in ("gemini", "google", "cloud"):
        return False
    return not bool(api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def get_chroma() -> chromadb.ClientAPI:
    """Get or create a persistent ChromaDB client."""
    global _chroma
    if _chroma is None:
        _chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    return _chroma


def embed_image(image_bytes: bytes, mime_type: str = "image/png", api_key: str | None = None) -> list[float]:
    """Embed an image using EmbeddingGemma 2 (local) or Gemini Multimodal Embeddings and return the embedding vector."""
    if _use_local_embeddinggemma(api_key):
        model = _get_embeddinggemma_model()
        with Image.open(io.BytesIO(image_bytes)) as img:
            vec = model.encode(img.convert("RGB"), normalize_embeddings=True)
            return vec.tolist() if hasattr(vec, "tolist") else list(vec)

    client = _get_client(api_key=api_key)
    target_model = GEMINI_EMBED_MODEL if "embeddinggemma" in MODEL else MODEL
    result = client.models.embed_content(
        model=target_model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
    )
    return result.embeddings[0].values


def embed_text(text: str, api_key: str | None = None, prompt_name: str = "SearchQuery") -> list[float]:
    """Embed a text query using EmbeddingGemma 2 (local) or Gemini Multimodal Embeddings and return the embedding vector."""
    if _use_local_embeddinggemma(api_key):
        model = _get_embeddinggemma_model()
        vec = model.encode(text, prompt_name=prompt_name, normalize_embeddings=True)
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)

    client = _get_client(api_key=api_key)
    target_model = GEMINI_EMBED_MODEL if "embeddinggemma" in MODEL else MODEL
    result = client.models.embed_content(
        model=target_model,
        contents=[text],
    )
    return result.embeddings[0].values


_npz_cache: dict[str, Any] | None = None
NPZ_FILE = os.path.join("embeddings", "comics.npz")


def _get_or_build_npz_index() -> dict[str, Any]:
    """Load or build a lightweight numpy (.npz) embedding cache using EmbeddingGemma 2 (no ChromaDB required)."""
    global _npz_cache
    if _npz_cache is not None:
        return _npz_cache

    import numpy as np

    with open(os.path.join("comics", "metadata.json")) as f:
        comics = json.load(f)
    meta_by_id = {str(c["num"]): c for c in comics}

    if os.path.exists(NPZ_FILE):
        data = np.load(NPZ_FILE)
        _npz_cache = {
            "ids": [str(x) for x in data["ids"].tolist()],
            "text_embeddings": data["text_embeddings"],
            "image_embeddings": data["image_embeddings"] if "image_embeddings" in data else data["text_embeddings"],
            "meta_by_id": meta_by_id,
        }
        return _npz_cache

    model = _get_embeddinggemma_model()
    ids = []
    docs = []
    for c in comics:
        cid = str(c["num"])
        title = c.get("title", "") or "none"
        body = c.get("transcript") or c.get("explanation") or ""
        ids.append(cid)
        docs.append(f"title: {title} | text: {body[:1500]}")

    txt_vecs = model.encode(docs, prompt_name="Document", normalize_embeddings=True)
    os.makedirs("embeddings", exist_ok=True)
    np.savez_compressed(NPZ_FILE, ids=np.array(ids), text_embeddings=txt_vecs)
    _npz_cache = {
        "ids": ids,
        "text_embeddings": txt_vecs,
        "image_embeddings": txt_vecs,
        "meta_by_id": meta_by_id,
    }
    return _npz_cache


def search(
    query_embedding: list[float],
    query_type: str = "image",
    top_k: int = 5,
) -> list[dict]:
    """Search via ChromaDB if indexed, or direct cosine similarity over EmbeddingGemma 2 vectors (comics.npz)."""
    chroma = get_chroma()

    # Query image collection if present; otherwise use direct numpy cosine similarity
    try:
        img_col = chroma.get_collection(IMAGE_COLLECTION)
    except Exception:
        import numpy as np

        idx = _get_or_build_npz_index()
        q_vec = np.asarray(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm
        sims = np.dot(idx["text_embeddings"], q_vec)
        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx_pos in top_indices:
            doc_id = idx["ids"][int(idx_pos)]
            meta = idx["meta_by_id"].get(doc_id, {})
            results.append({
                "comic_id": int(doc_id),
                "score": round(float(sims[int(idx_pos)]), 4),
                "title": meta.get("title", ""),
                "transcript": meta.get("transcript", ""),
                "explanation": meta.get("explanation", ""),
                "filename": meta.get("filename", ""),
                "url": f"https://xkcd.com/{doc_id}/",
            })
        return results

    img_results = img_col.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "distances"],
    )

    # Build scores dict: comic_id -> {score, metadata}
    scores: dict[str, dict] = {}
    for i, doc_id in enumerate(img_results["ids"][0]):
        # ChromaDB cosine distance = 1 - similarity
        sim = 1.0 - img_results["distances"][0][i]
        meta = img_results["metadatas"][0][i]
        scores[doc_id] = {"score": sim, "metadata": meta}

    # For text queries, also query text collection and take max
    if query_type == "text":
        try:
            txt_col = chroma.get_collection(TEXT_COLLECTION)
            txt_results = txt_col.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["metadatas", "distances"],
            )
            for i, doc_id in enumerate(txt_results["ids"][0]):
                txt_sim = 1.0 - txt_results["distances"][0][i]
                if doc_id in scores:
                    scores[doc_id]["score"] = max(scores[doc_id]["score"], txt_sim)
                else:
                    meta = txt_results["metadatas"][0][i]
                    scores[doc_id] = {"score": txt_sim, "metadata": meta}
        except Exception:
            pass  # text collection might not exist in older indexes

    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)[:top_k]

    results = []
    for doc_id, data in ranked:
        meta = data["metadata"]
        results.append({
            "comic_id": int(doc_id),
            "score": round(data["score"], 4),
            "title": meta.get("title", ""),
            "transcript": meta.get("transcript", ""),
            "explanation": meta.get("explanation", ""),
            "filename": meta.get("filename", ""),
            "url": f"https://xkcd.com/{doc_id}/",
        })

    return results


COMICS_DIR = "comics"
METADATA_FILE = os.path.join(COMICS_DIR, "metadata.json")
TYPESAFE_MODEL = os.getenv("TYPESAFE_MODEL", "jev-latest")

_comics_metadata: list[dict] | None = None
_typesafe_async_client: AsyncTypeSafeClient | None = None


def get_typesafe_client() -> AsyncTypeSafeClient:
    """Get or create reusable AsyncTypeSafeClient with connection pooling."""
    global _typesafe_async_client
    if _typesafe_async_client is None:
        _typesafe_async_client = AsyncTypeSafeClient()
    return _typesafe_async_client


def get_comics_metadata() -> list[dict]:
    """Load and cache comics metadata from disk."""
    global _comics_metadata
    if _comics_metadata is None:
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE) as f:
                _comics_metadata = json.load(f)
        else:
            _comics_metadata = []
    return _comics_metadata


async def search_typesafe(
    query: str,
    top_k: int = 5,
    model: str = TYPESAFE_MODEL,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Search XKCD comics directly using TypeSafe AI System One (Jev).

    Uses a Choice question to evaluate all candidate comics in a single call,
    accompanied by a Noul question to check whether a plausible match exists.
    Returns calibrated probabilities, confidence score, and execution latency.
    """
    comics = get_comics_metadata()
    if not comics:
        raise ValueError("No comics metadata found. Run fetch_xkcd.py first.")

    # Choice allows up to 255 options. If comics > 250, use the top 250 slice.
    candidates_pool = comics[:250]
    criteria = {}
    comic_by_id = {}
    for c in candidates_pool:
        cid = str(c["num"])
        comic_by_id[cid] = c
        preview = (c.get("transcript") or c.get("explanation") or "")[:180].replace("\n", " ")
        criteria[cid] = f"{c['title']}: {preview}"

    t0 = time.perf_counter()
    async with AsyncTypeSafeClient(api_key=api_key) as client:
        resp = await client.system_one(
            state={"query": query},
            questions={
                "has_match": Noul(
                    instructions=f'Does any XKCD comic in the candidate list plausibly match this query: "{query}"?'
                ),
                "match": Choice(
                    instructions=f'Which XKCD comic best matches the user query: "{query}"?',
                    criteria=criteria,
                ),
            },
            model=model,
        )
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000

    match_ans = resp.answers["match"]
    probs = match_ans.probabilities
    has_match_noul = resp.answers["has_match"].noul

    ranked_ids = sorted(probs.keys(), key=lambda cid: probs[cid], reverse=True)[:top_k]

    results = []
    for cid in ranked_ids:
        c = comic_by_id.get(cid)
        if not c:
            continue
        prob = probs[cid]
        results.append({
            "comic_id": int(cid),
            "score": round(prob, 4),
            "confidence": round(match_ans.confidence, 4) if cid == match_ans.choice else None,
            "title": c.get("title", ""),
            "transcript": c.get("transcript", ""),
            "explanation": c.get("explanation", ""),
            "filename": c.get("filename", ""),
            "url": f"https://xkcd.com/{cid}/",
        })

    return {
        "results": results,
        "engine": "typesafe",
        "model": resp.model or model,
        "latency_ms": round(latency_ms, 1),
        "has_match_noul": round(has_match_noul, 4),
        "input_tokens": resp.usage.input_tokens if resp.usage else None,
        "output_tokens": resp.usage.output_tokens if resp.usage else None,
    }


async def rerank_with_typesafe(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    model: str = TYPESAFE_MODEL,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Re-rank candidate results from fast search (e.g. ChromaDB) using TypeSafe Jev."""
    if not candidates:
        return {"results": [], "latency_ms": 0.0, "engine": "rerank", "model": model}

    criteria = {}
    for c in candidates:
        cid = str(c["comic_id"])
        preview = (c.get("transcript") or c.get("explanation") or "")[:180].replace("\n", " ")
        criteria[cid] = f"{c.get('title', '')}: {preview}"

    t0 = time.perf_counter()
    async with AsyncTypeSafeClient(api_key=api_key) as client:
        resp = await client.system_one(
            state={"query": query},
            questions={
                "has_match": Noul(
                    instructions=f'Does any candidate plausibly match this query: "{query}"?'
                ),
                "match": Choice(
                    instructions=f'Which candidate XKCD comic best matches the user query: "{query}"?',
                    criteria=criteria,
                ),
            },
            model=model,
        )
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000

    match_ans = resp.answers["match"]
    probs = match_ans.probabilities
    has_match_noul = resp.answers["has_match"].noul

    reranked_list = []
    for c in candidates:
        item = dict(c)
        cid = str(item["comic_id"])
        item["typesafe_prob"] = round(probs.get(cid, 0.0), 4)
        item["vector_score"] = item.get("score", 0.0)
        item["score"] = item["typesafe_prob"]
        if cid == match_ans.choice:
            item["confidence"] = round(match_ans.confidence, 4)
        reranked_list.append(item)

    reranked = sorted(reranked_list, key=lambda x: x["score"], reverse=True)[:top_k]

    return {
        "results": reranked,
        "engine": "rerank",
        "model": resp.model or model,
        "latency_ms": round(latency_ms, 1),
        "has_match_noul": round(has_match_noul, 4),
        "input_tokens": resp.usage.input_tokens if resp.usage else None,
        "output_tokens": resp.usage.output_tokens if resp.usage else None,
    }
