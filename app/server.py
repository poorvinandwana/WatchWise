import os
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import core

import logging

print("=== SERVER: imports complete ===", flush=True)

logger = logging.getLogger(__name__)

print("=== SERVER: logger created ===", flush=True)

app = FastAPI(title="WatchWise")

print("=== SERVER: FastAPI app created ===", flush=True)


app = FastAPI(title="WatchWise")

# Loaded once on first use, not per-request — the embedding model and Chroma
# connection are relatively expensive to set up.
_genai_client = None
_embedder = None
_collection = None


def get_clients():
    global _genai_client, _embedder, _collection

    print("=== STEP 1: get_clients() entered ===", flush=True)

    if _embedder is None:
        print("=== STEP 2: loading embedder ===", flush=True)
        _embedder = core.get_embedder()
        print("=== STEP 3: embedder loaded ===", flush=True)
    else:
        print("=== STEP 3A: embedder already cached ===", flush=True)

    if _collection is None:
        print("=== STEP 4: loading collection ===", flush=True)
        _collection = core.get_collection()
        print("=== STEP 5: collection loaded ===", flush=True)
    else:
        print("=== STEP 5A: collection already cached ===", flush=True)

    print("=== STEP 6: get_clients() returning ===", flush=True)
    return _embedder, _collection


def get_genai_client():
    global _genai_client

    print("=== STEP G1: get_genai_client() ===", flush=True)

    if _genai_client is None:
        print("=== STEP G2: creating Gemini client ===", flush=True)
        _genai_client = core.get_genai_client()
        print("=== STEP G3: Gemini client ready ===", flush=True)
    else:
        print("=== STEP G3A: Gemini client cached ===", flush=True)

    return _genai_client

class QueryRequest(BaseModel):
    question: str
    suspicious_only: bool = False

@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)):
    print("=== /api/ingest called ===", flush=True)

    embedder, collection = get_clients()
    print("=== embedder + collection acquired ===", flush=True)

    genai_client = get_genai_client()
    print("=== Gemini client acquired ===", flush=True)

    suffix = os.path.splitext(file.filename)[1] or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        print("=== calling ingest_clip() ===", flush=True)

        result = core.ingest_clip(
            genai_client,
            embedder,
            collection,
            tmp_path,
            display_filename=file.filename,
        )

        print("=== ingest_clip() finished ===", flush=True)

    except Exception:
        logger.exception("Ingestion failed")
        raise HTTPException(
            status_code=500,
            detail="Video processing failed."
        )
    finally:
        os.remove(tmp_path)

    return result

@app.get("/api/events")
async def list_events():
    _, collection = get_clients()
    data = collection.get()
    events = []
    for i, meta in enumerate(data["metadatas"]):
        events.append({**meta, "description": data["documents"][i]})
    events.sort(key=lambda e: e["ingestion_timestamp"], reverse=True)
    return events

@app.get("/api/status")
async def status():
    _, collection = get_clients()

    return {
        "indexed": collection.count() > 0,
        "event_count": collection.count(),
    }


@app.post("/api/query")
async def query(req: QueryRequest):
    print("=== /api/query called ===", flush=True)
    embedder, collection = get_clients()
    try:
        print("=== calling query_events() ===", flush=True)
        result = core.query_events(embedder, collection, req.question, suspicious_only=req.suspicious_only)
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(
            status_code=500,
            detail="Query failed."
        )
    return result


# Serve the frontend. Must be mounted last so it doesn't shadow the /api routes above.
from pathlib import Path

STATIC_DIR = Path(__file__).parent / "static"

app.mount(
    "/",
    StaticFiles(directory=str(STATIC_DIR), html=True),
    name="static",
)
