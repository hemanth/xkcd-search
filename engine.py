"""Embedding engine using gemini-embedding-2-preview for multimodal search."""

import os
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not set")
        _client = genai.Client(api_key=api_key)
    return _client


MODEL = "gemini-embedding-2-preview"


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


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def load_index(path: str = "embeddings/comics.npz") -> dict:
    """Load pre-computed embeddings. Returns dict with ids, image_embeddings, text_embeddings."""
    data = np.load(path)
    result = {
        "ids": data["ids"].tolist(),
    }
    # Support both old (single 'embeddings') and new (split) formats
    if "image_embeddings" in data:
        result["image_embeddings"] = data["image_embeddings"]
        result["text_embeddings"] = data["text_embeddings"]
    else:
        result["image_embeddings"] = data["embeddings"]
        result["text_embeddings"] = None
    return result


def search(
    query_embedding: list[float],
    index: dict,
    top_k: int = 5,
    query_type: str = "image",
) -> list[dict]:
    """Search for the closest comics using hybrid scoring.

    For text queries: scores against both image and text indexes, takes the max.
    For image queries: scores against image index only.
    """
    query_vec = np.array(query_embedding)
    ids = index["ids"]
    image_matrix = index["image_embeddings"]
    text_matrix = index.get("text_embeddings")

    scores = []
    for i, cid in enumerate(ids):
        img_sim = cosine_similarity(query_vec, image_matrix[i])

        if query_type == "text" and text_matrix is not None:
            txt_sim = cosine_similarity(query_vec, text_matrix[i])
            # For text queries, use the higher of the two scores
            sim = max(img_sim, txt_sim)
        else:
            sim = img_sim

        scores.append((cid, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [{"comic_id": cid, "score": round(score, 4)} for cid, score in scores[:top_k]]
