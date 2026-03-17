"""Embedding engine using gemini-embedding-2-preview + ChromaDB for multimodal search."""

import os

import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

_client = None
_chroma = None

MODEL = "gemini-embedding-2-preview"
CHROMA_DIR = "chroma_db"
IMAGE_COLLECTION = "xkcd_images"
TEXT_COLLECTION = "xkcd_text"


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not set")
        _client = genai.Client(api_key=api_key)
    return _client


def get_chroma() -> chromadb.ClientAPI:
    """Get or create a persistent ChromaDB client."""
    global _chroma
    if _chroma is None:
        _chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    return _chroma


def embed_image(image_bytes: bytes, mime_type: str = "image/png") -> list[float]:
    """Embed an image and return the embedding vector."""
    client = _get_client()
    result = client.models.embed_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
    )
    return result.embeddings[0].values


def embed_text(text: str) -> list[float]:
    """Embed a text query and return the embedding vector."""
    client = _get_client()
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
    img_col = chroma.get_collection(IMAGE_COLLECTION)
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
