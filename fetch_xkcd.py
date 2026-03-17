"""Load XKCD comics from the Hugging Face dataset olivierdehaene/xkcd."""

import json
import os
import sys
import time

import httpx

COMICS_DIR = "comics"
IMAGES_DIR = os.path.join(COMICS_DIR, "images")
METADATA_FILE = os.path.join(COMICS_DIR, "metadata.json")

HF_DATASET = "olivierdehaene/xkcd"


def download_image(url: str, dest: str) -> bool:
    """Download an image to disk."""
    try:
        resp = httpx.get(url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            f.write(resp.content)
        return True
    except Exception as e:
        print(f"  ⚠ failed to download {url}: {e}")
        return False


def load_dataset(count: int = 50):
    """Load XKCD comics from HF dataset and download images."""
    from datasets import load_dataset

    os.makedirs(IMAGES_DIR, exist_ok=True)

    print(f"📦 Loading HF dataset '{HF_DATASET}'…")
    ds = load_dataset(HF_DATASET, split="train")

    # Take a slice — last N comics by id
    ds = ds.sort("id", reverse=True)
    subset = ds.select(range(min(count, len(ds))))

    metadata = []
    total = len(subset)

    for i, row in enumerate(subset):
        num = row["id"]
        title = row.get("title", "")
        img_url = row.get("image_url", "")
        transcript = row.get("transcript") or ""
        explanation = row.get("explanation") or ""

        if not img_url:
            continue

        ext = os.path.splitext(img_url.split("?")[0])[-1] or ".png"
        filename = f"{num}{ext}"
        dest = os.path.join(IMAGES_DIR, filename)

        print(f"  [{i + 1}/{total}] #{num}: {title}")

        if not os.path.exists(dest):
            if not download_image(img_url, dest):
                continue
            time.sleep(0.15)

        metadata.append(
            {
                "num": num,
                "title": title,
                "transcript": transcript,
                "explanation": explanation,
                "img_url": img_url,
                "filename": filename,
            }
        )

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Saved {len(metadata)} comics to {METADATA_FILE}")


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    load_dataset(count)
