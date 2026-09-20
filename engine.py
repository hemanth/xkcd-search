import json
import os
import time
from typing import Any

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul

load_dotenv()


_client = None
_chroma = None

MODEL = "gemini-embedding-2-preview"
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


def get_chroma() -> chromadb.ClientAPI:
    """Get or create a persistent ChromaDB client."""
    global _chroma
    if _chroma is None:
        _chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    return _chroma


def embed_image(image_bytes: bytes, mime_type: str = "image/png", api_key: str | None = None) -> list[float]:
    """Embed an image and return the embedding vector."""
    client = _get_client(api_key=api_key)
    result = client.models.embed_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
    )
    return result.embeddings[0].values


def embed_text(text: str, api_key: str | None = None) -> list[float]:
    """Embed a text query and return the embedding vector."""
    client = _get_client(api_key=api_key)
    result = client.models.embed_content(
        model=MODEL,
        contents=[text],
    )
    return result.embeddings[0].values


def search(
    query_embedding: list[float],
    query_type: str = "image",
    top_k: int = 5,
) -> list[dict]:
    """Search ChromaDB collections using hybrid scoring.

    For text queries: queries both image and text collections, takes the max score.
    For image queries: queries image collection only.
    """
    chroma = get_chroma()

    # Query image collection
    try:
        img_col = chroma.get_collection(IMAGE_COLLECTION)
    except Exception as e:
        raise ValueError(
            "ChromaDB collection [xkcd_images] does not exist yet. Run 'python index_comics.py' to build the vector embeddings, or switch to 'TypeSafe Jev' which works immediately."
        ) from e

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
