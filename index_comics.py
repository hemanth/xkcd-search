"""Pre-compute embeddings for all scraped XKCD comics (image + text)."""

import json
import os
import sys
import time

import numpy as np

from engine import embed_image, embed_text

COMICS_DIR = "comics"
IMAGES_DIR = os.path.join(COMICS_DIR, "images")
METADATA_FILE = os.path.join(COMICS_DIR, "metadata.json")
EMBEDDINGS_DIR = "embeddings"
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "comics.npz")


def index_comics():
    """Embed every comic image AND its text (title + transcript) and save to disk."""
    with open(METADATA_FILE) as f:
        comics = json.load(f)

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    ids = []
    image_embeddings = []
    text_embeddings = []
    total = len(comics)

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

        try:
            print(f"  [{i + 1}/{total}] Embedding #{num}: {title}")

            # Image embedding
            img_vec = embed_image(image_bytes, mime_type)
            time.sleep(0.25)

            # Text embedding
            txt_vec = embed_text(text_blob[:2000])  # keep within limits
            time.sleep(0.25)

            ids.append(num)
            image_embeddings.append(img_vec)
            text_embeddings.append(txt_vec)
        except Exception as e:
            print(f"  ⚠ Failed #{num}: {e}")
            continue

    np.savez(
        EMBEDDINGS_FILE,
        ids=np.array(ids, dtype=np.int32),
        image_embeddings=np.array(image_embeddings, dtype=np.float32),
        text_embeddings=np.array(text_embeddings, dtype=np.float32),
    )
    print(f"✅ Indexed {len(ids)} comics (image + text) → {EMBEDDINGS_FILE}")


if __name__ == "__main__":
    index_comics()
