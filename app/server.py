import os
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import core

import logging

logger = logging.getLogger(__name__)


app = FastAPI(title="WatchWise")

# Loaded once on first use, not per-request — the embedding model and Chroma
# connection are relatively expensive to set up.
_genai_client = None
_embedder = None
_collection = None


def get_clients():
    global _genai_client, _embedder, _collection
    if _embedder is None:
        _embedder = core.get_embedder()
        _collection = core.get_collection()
    return _embedder, _collection


def get_genai_client():
    global _genai_client
    if _genai_client is None:
        _genai_client = core.get_genai_client()
    return _genai_client


class QueryRequest(BaseModel):
    question: str
    suspicious_only: bool = False


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)):
    embedder, collection = get_clients()
    genai_client = get_genai_client()

    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = core.ingest_clip(
            genai_client, embedder, collection, tmp_path, display_filename=file.filename
        )
    except Exception as exc:
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
    embedder, collection = get_clients()
    try:
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
