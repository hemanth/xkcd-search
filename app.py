"""FastAPI server for XKCD reverse image lookup."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from engine import embed_image, embed_text, get_chroma, search, IMAGE_COLLECTION

IMAGES_DIR = os.path.join("comics", "images")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔍 Loading XKCD index…")
    try:
        chroma = get_chroma()
        img_col = chroma.get_collection(IMAGE_COLLECTION)
        count = img_col.count()
        print(f"✅ ChromaDB loaded — {count} comics indexed")
    except Exception as e:
        print(f"⚠️  Could not load index: {e}")
        print("   Run index_comics.py first.")
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

        results = search(embedding, query_type=query_type, top_k=5)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return {"results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
