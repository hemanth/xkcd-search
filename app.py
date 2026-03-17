"""FastAPI server for XKCD reverse image lookup."""

import json
import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from engine import embed_image, embed_text, load_index, search

COMICS_DIR = "comics"
METADATA_FILE = os.path.join(COMICS_DIR, "metadata.json")
IMAGES_DIR = os.path.join(COMICS_DIR, "images")

# Loaded at startup
comic_index: dict | None = None
comics_meta: dict[int, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global comic_index, comics_meta
    print("🔍 Loading XKCD index…")
    try:
        index = load_index()
        if len(index["ids"]) > 0:
            comic_index = index
            has_text = index.get("text_embeddings") is not None
            print(f"✅ Loaded {len(index['ids'])} embeddings (hybrid={'yes' if has_text else 'no'})")
        else:
            print("⚠️  Index is empty — run index_comics.py with a valid API key")
    except Exception as e:
        print(f"⚠️  Could not load index: {e}")

    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE) as f:
            for c in json.load(f):
                comics_meta[c["num"]] = c
        print(f"📚 Loaded {len(comics_meta)} comics metadata")
    yield


app = FastAPI(title="XKCD Reverse Lookup", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("static/index.html")


@app.get("/comics/images/{filename}")
async def serve_comic_image(filename: str):
    path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path)


@app.post("/api/search")
async def search_comics(
    file: UploadFile | None = File(None),
    query: str | None = Form(None),
):
    if comic_index is None:
        return JSONResponse(
            {"error": "Index not loaded — run index_comics.py with a valid API key"},
            status_code=503,
        )

    try:
        if file and file.filename:
            image_bytes = await file.read()
            mime = file.content_type or "image/png"
            embedding = embed_image(image_bytes, mime)
            query_type = "image"
        elif query:
            embedding = embed_text(query)
            query_type = "text"
        else:
            return JSONResponse({"error": "Provide an image or text query"}, status_code=400)

        results = search(embedding, comic_index, top_k=5, query_type=query_type)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    # Enrich with metadata
    for r in results:
        meta = comics_meta.get(r["comic_id"], {})
        r["title"] = meta.get("title", "")
        r["transcript"] = meta.get("transcript", "")
        r["explanation"] = meta.get("explanation", "")
        r["filename"] = meta.get("filename", "")
        r["url"] = f"https://xkcd.com/{r['comic_id']}/"

    return {"results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
