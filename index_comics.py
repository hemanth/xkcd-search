"""Pre-compute embeddings for all scraped XKCD comics and store in ChromaDB."""

import json
import os
import sys
import time

from engine import embed_image, embed_text, get_chroma, IMAGE_COLLECTION, TEXT_COLLECTION

COMICS_DIR = "comics"
IMAGES_DIR = os.path.join(COMICS_DIR, "images")
METADATA_FILE = os.path.join(COMICS_DIR, "metadata.json")


def index_comics():
    """Embed every comic image AND its text, store in ChromaDB."""
    with open(METADATA_FILE) as f:
        comics = json.load(f)

    chroma = get_chroma()

    # Create or reset collections with cosine distance
    try:
        chroma.delete_collection(IMAGE_COLLECTION)
    except Exception:
        pass
    try:
        chroma.delete_collection(TEXT_COLLECTION)
    except Exception:
        pass

    img_col = chroma.create_collection(
        name=IMAGE_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    txt_col = chroma.create_collection(
        name=TEXT_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    total = len(comics)
    indexed = 0

    for i, comic in enumerate(comics):
        num = comic["num"]
        path = os.path.join(IMAGES_DIR, comic["filename"])
        if not os.path.exists(path):
            print(f"  ⚠ Skipping #{num}: image not found")
            continue

        ext = os.path.splitext(comic["filename"])[-1].lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif"}
        mime_type = mime_map.get(ext, "image/png")

        with open(path, "rb") as f:
            image_bytes = f.read()

        # Build text blob from title + transcript
        title = comic.get("title", "")
        transcript = comic.get("transcript", "")
        explanation = comic.get("explanation", "")
        text_blob = f"{title}. {transcript}".strip()
        if not text_blob or text_blob == ".":
            text_blob = f"{title}. {explanation}".strip()

        # Metadata to store alongside embeddings
        meta = {
            "title": title,
            "transcript": transcript[:4000],       # ChromaDB metadata size limit
            "explanation": explanation[:4000],
            "filename": comic["filename"],
        }

        try:
            print(f"  [{i + 1}/{total}] Embedding #{num}: {title}")
            doc_id = str(num)

            # Image embedding
            img_vec = embed_image(image_bytes, mime_type)
            img_col.add(ids=[doc_id], embeddings=[img_vec], metadatas=[meta])
            time.sleep(0.25)

            # Text embedding
            txt_vec = embed_text(text_blob[:2000])
            txt_col.add(ids=[doc_id], embeddings=[txt_vec], metadatas=[meta])
            time.sleep(0.25)

            indexed += 1
        except Exception as e:
            print(f"  ⚠ Failed #{num}: {e}")
            continue

    print(f"✅ Indexed {indexed} comics in ChromaDB (image + text collections)")


if __name__ == "__main__":
    index_comics()
