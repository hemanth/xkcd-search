"""FastAPI server for XKCD reverse lookup with Gemini Multimodal Embeddings + TypeSafe AI."""

import asyncio
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from engine import (
    embed_image,
    embed_text,
    get_chroma,
    get_comics_metadata,
    search,
    search_typesafe,
    rerank_with_typesafe,
    IMAGE_COLLECTION,
    TYPESAFE_MODEL,
)

IMAGES_DIR = os.path.join("comics", "images")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing XKCD search engine...")
    comics = get_comics_metadata()
    print(f"TypeSafe AI ready: {len(comics)} comics loaded in metadata")

    try:
        chroma = get_chroma()
        img_col = chroma.get_collection(IMAGE_COLLECTION)
        count = img_col.count()
        print(f"ChromaDB loaded: {count} comics indexed")
    except Exception as e:
        print(f"ChromaDB not indexed ({e}). TypeSafe AI search is fully active.")

    yield


app = FastAPI(title="XKCD Reverse Lookup", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def root():
    return FileResponse("static/index.html")


@app.get("/comics/images/{filename}")
async def serve_comic_image(filename: str):
    path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path)


@app.get("/api/keys/status")
async def get_keys_status():
    """Report whether server environment has default keys configured and Chroma index status."""
    ts_key = os.getenv("TYPESAFE_API_KEY")
    gm_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    chroma_indexed = False
    try:
        chroma = get_chroma()
        colls = [c.name for c in chroma.list_collections()]
        chroma_indexed = IMAGE_COLLECTION in colls
    except Exception:
        pass

    return {
        "typesafe_configured": bool(ts_key),
        "typesafe_masked": (ts_key[:10] + "..." + ts_key[-4:]) if ts_key else None,
        "gemini_configured": bool(gm_key),
        "gemini_masked": (gm_key[:8] + "..." + gm_key[-4:]) if gm_key else None,
        "chroma_indexed": chroma_indexed,
    }


@app.post("/api/search")
async def search_comics(
    request: Request,
    file: UploadFile | None = File(None),
    query: str | None = Form(None),
    engine: str = Form("typesafe"),
    top_k: int = Form(5),
    model: str | None = Form(None),
    typesafe_api_key: str | None = Form(None),
    gemini_api_key: str | None = Form(None),
):
    """Search XKCD comics with selected engine:

    - typesafe: TypeSafe System One (Jev) direct semantic choice & calibrated probability.
    - chroma: Gemini Multimodal Embedding (gemini-embedding-2-preview) + ChromaDB vector search.
    - rerank: Gemini + ChromaDB vector retrieval re-ranked by TypeSafe Jev.
    - auto: image -> chroma; text -> typesafe.
    """
    try:
        ts_model = model or TYPESAFE_MODEL

        # Priority: explicit form parameter -> request header -> server env variable
        ts_key = typesafe_api_key or request.headers.get("x-typesafe-api-key") or os.getenv("TYPESAFE_API_KEY")
        gm_key = gemini_api_key or request.headers.get("x-gemini-api-key") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        # Resolve auto engine
        if engine == "auto":
            engine = "chroma" if (file and file.filename) else "typesafe"

        # 1. Image upload search (Gemini Multimodal Embedding)
        if file and file.filename:
            if not gm_key:
                return JSONResponse({
                    "error": "Gemini API key is required for image search. Please add your key in the API Keys menu or set GEMINI_API_KEY."
                }, status_code=400)

            image_bytes = await file.read()
            mime = file.content_type or "image/png"
            t0 = time.perf_counter()
            embedding = embed_image(image_bytes, mime, api_key=gm_key)
            retrieval_k = top_k if engine != "rerank" else max(top_k * 3, 10)
            try:
                results = search(embedding, query_type="image", top_k=retrieval_k)
            except ValueError:
                return JSONResponse({
                    "error": "ChromaDB collection [xkcd_images] does not exist yet. Run 'python index_comics.py' to generate image embeddings, or use Text Search with TypeSafe Jev."
                }, status_code=400)
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000

            if engine == "rerank" and query:
                rerank_resp = await rerank_with_typesafe(query, results, top_k=top_k, model=ts_model, api_key=ts_key)
                rerank_resp["total_latency_ms"] = round(latency_ms + rerank_resp["latency_ms"], 1)
                return rerank_resp

            return {
                "results": results[:top_k],
                "engine": "chroma",
                "model": "gemini-embedding-2-preview",
                "latency_ms": round(latency_ms, 1),
            }

        # 2. Text search
        if not query or not query.strip():
            return JSONResponse({"error": "Provide an image or text query"}, status_code=400)

        query = query.strip()

        if engine == "typesafe":
            if not ts_key:
                return JSONResponse({
                    "error": "TypeSafe API key is required. Please enter your key in the API Keys menu or set TYPESAFE_API_KEY."
                }, status_code=400)

            resp = await search_typesafe(query, top_k=top_k, model=ts_model, api_key=ts_key)
            return resp

        elif engine == "chroma":
            if not gm_key:
                return JSONResponse({
                    "error": "Gemini API key is required for Gemini + Chroma search. Please set your key in API Keys or switch to TypeSafe Jev."
                }, status_code=400)

            t0 = time.perf_counter()
            embedding = embed_text(query, api_key=gm_key)
            try:
                results = search(embedding, query_type="text", top_k=top_k)
            except ValueError:
                return JSONResponse({
                    "error": "ChromaDB collection [xkcd_images] does not exist yet. Run 'python index_comics.py' with GEMINI_API_KEY to build it, or switch to TypeSafe Jev above which searches instantly without an index."
                }, status_code=400)
            t1 = time.perf_counter()
            return {
                "results": results,
                "engine": "chroma",
                "model": "gemini-embedding-2-preview",
                "latency_ms": round((t1 - t0) * 1000, 1),
            }

        elif engine == "rerank":
            if not gm_key:
                # If Gemini key not available, seamlessly fall back to direct TypeSafe search
                resp = await search_typesafe(query, top_k=top_k, model=ts_model, api_key=ts_key)
                return resp

            try:
                t0 = time.perf_counter()
                embedding = embed_text(query, api_key=gm_key)
                candidates = search(embedding, query_type="text", top_k=max(top_k * 3, 12))
                t_chroma = (time.perf_counter() - t0) * 1000

                rerank_resp = await rerank_with_typesafe(query, candidates, top_k=top_k, model=ts_model, api_key=ts_key)
                rerank_resp["chroma_latency_ms"] = round(t_chroma, 1)
                rerank_resp["total_latency_ms"] = round(t_chroma + rerank_resp["latency_ms"], 1)
                return rerank_resp
            except ValueError:
                # Chroma index missing -> seamlessly fall back to direct TypeSafe Jev search
                resp = await search_typesafe(query, top_k=top_k, model=ts_model, api_key=ts_key)
                return resp

        else:
            return JSONResponse({"error": f"Unknown engine: {engine}"}, status_code=400)

    except Exception as e:
        error_msg = str(e)
        if "GEMINI_API_KEY" in error_msg or "GOOGLE_API_KEY" in error_msg:
            error_msg += ". Tip: Switch to 'TypeSafe Jev' to search immediately without Gemini embeddings, or enter your key in API Keys."
        return JSONResponse({"error": error_msg}, status_code=500)


@app.get("/api/benchmark")
async def run_benchmark(
    request: Request,
    model: str = TYPESAFE_MODEL,
    typesafe_api_key: str | None = None,
):
    """Run a live latency benchmark across sample queries using TypeSafe AI."""
    ts_key = typesafe_api_key or request.headers.get("x-typesafe-api-key") or os.getenv("TYPESAFE_API_KEY")
    if not ts_key:
        return JSONResponse({"error": "TypeSafe API key not configured."}, status_code=400)

    benchmark_queries = [
        "the space shuttle is actually a mammal with bones",
        "vibrating rapidly to create blur in fast shutter speed photos",
        "rolling a die with sixty-five thousand five hundred thirty-six sides",
        "sears tower was the world's tallest tower or building in the 90s",
        "regular and electron types of scopes like telescope and kaleidoscope",
    ]

    comics = get_comics_metadata()
    if not comics:
        return JSONResponse({"error": "No comics loaded. Run fetch_xkcd.py first."}, status_code=400)

    results = []
    latencies = []

    for q in benchmark_queries:
        resp = await search_typesafe(q, top_k=1, model=model, api_key=ts_key)
        ms = resp["latency_ms"]
        latencies.append(ms)
        top = resp["results"][0] if resp["results"] else {}
        results.append({
            "query": q,
            "latency_ms": ms,
            "top_match_id": top.get("comic_id"),
            "top_match_title": top.get("title"),
            "probability": top.get("score"),
            "confidence": top.get("confidence"),
            "has_match_noul": resp.get("has_match_noul"),
        })

    latencies.sort()
    return {
        "model": model,
        "queries_count": len(benchmark_queries),
        "min_latency_ms": latencies[0],
        "max_latency_ms": latencies[-1],
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
        "p50_latency_ms": latencies[len(latencies) // 2],
        "queries": results,
    }


@app.post("/api/index")
async def build_index(request: Request):
    """Build ChromaDB vector index using the provided Gemini API key."""
    gm_key = request.headers.get("x-gemini-api-key") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gm_key:
        return JSONResponse(
            {"error": "Gemini API key is required to build ChromaDB embeddings. Please enter your key in API Keys first."},
            status_code=400,
        )

    try:
        from index_comics import index_comics
        loop = asyncio.get_running_loop()
        indexed = await loop.run_in_executor(None, index_comics, gm_key)
        return {
            "success": True,
            "indexed": indexed,
            "message": f"Successfully indexed {indexed} comics into ChromaDB.",
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
