from .config import *
from pathlib import Path
import os
import time
import uuid
from datetime import datetime, timezone
from fastapi import FastAPI, File, HTTPException, UploadFile
import chromadb
from google import genai
from google.genai import types
SentenceTransformer = None
from groq import Groq

from .config import (
    MODEL,
    VIDEO_FPS,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GROQ_MODEL,
    TOP_K,
    HEDGE_WORDS,
)
STAGE1_PROMPT = """Watch this video clip carefully.

First, list every distinct object and every person visible at the very start of the clip \
(e.g. "laptop on lower bunk table", "phone held by person in striped shirt"), with their \
approximate starting location.

Then describe the clip in chronological order, tracking two things specifically:

1. For EACH object you listed: does it move, change hands, get picked up, or disappear from \
view at any point? If an object is no longer visible later in the clip, say so explicitly and \
note the last moment you saw it.
2. Does anything or anyone enter the frame from OUTSIDE the visible space — through a window, \
door, gap, or from off-screen — rather than just movement between people already in frame? \
Check the edges of the frame, not just the center/foreground action.

Do not assume the only possible action is between people already seated/present — actively \
check whether anything reaches in from outside the visible space.

Be purely descriptive and factual. Do not assess whether behavior is suspicious, illegal, \
or theft — only describe observable actions, movements, and the fate of each object."""

STAGE2_PROMPT_TEMPLATE = """Below is a factual, object-by-object description of a scene, generated from video footage:

"{description}"

Based only on this description, assess whether the events match patterns commonly associated \
with theft or unauthorized taking of an object — this includes an insider concealing an item, \
someone reaching in from outside the visible space (e.g. through a window or door) to take an \
object, or any object disappearing from view without a clear legitimate handoff.

Respond with:
- Classification: Suspicious / Not Suspicious / Insufficient Information
- Supporting details: which specific elements of the description support this classification
- Confidence: High / Medium / Low"""

SYNTHESIS_PROMPT_TEMPLATE = """You are answering questions about CCTV footage based on the retrieved event \
descriptions below. Each event includes a description, a classification, a confidence level, and whether it \
contains uncertain/hedged language.

Some retrieved event descriptions may contain uncertain or partial observations (e.g. "possibly a wallet", \
"object no longer visible"). When the source description itself expresses uncertainty about a detail, reflect \
that uncertainty in your answer rather than stating it as fact. Only state something as fact if the source \
description does.

Cite which event (by source filename) supports each part of your answer. If the retrieved events don't \
actually answer the question, say so rather than guessing.

Retrieved events:
{context}

Question: {question}

Answer:"""
import logging

logging.basicConfig(level=logging.INFO)

print("=== IMPORTED SERVER.PY ===", flush=True)

app = FastAPI(title="WatchWise")


@app.on_event("startup")
async def startup():
    print("=== FASTAPI STARTUP ===", flush=True)

def contains_hedge_language(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in HEDGE_WORDS)


def normalize_classification(raw: str) -> str:
    """Collapse Gemini's free-text classification into one of three exact labels,
    so metadata filtering (e.g. suspicious-only queries) doesn't silently miss
    entries due to punctuation or phrasing variance."""
    lowered = raw.lower()
    if "not suspicious" in lowered:
        return "Not Suspicious"
    if "insufficient" in lowered:
        return "Insufficient Information"
    if "suspicious" in lowered:
        return "Suspicious"
    return "Insufficient Information"


def get_genai_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)

_embedder = None

def get_embedder():
    global _embedder

    if _embedder is None:
        print(">>> get_embedder(): importing SentenceTransformer", flush=True)
        from sentence_transformers import SentenceTransformer

        print(">>> get_embedder(): import complete", flush=True)

        print(f">>> get_embedder(): loading {EMBEDDING_MODEL}", flush=True)
        _embedder = SentenceTransformer(EMBEDDING_MODEL)

        print(">>> get_embedder(): model loaded", flush=True)

    return _embedder

def get_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return chroma_client.get_or_create_collection(COLLECTION_NAME)


def upload_and_wait(genai_client, video_path):
    video_file = genai_client.files.upload(file=video_path)
    while video_file.state.name == "PROCESSING":
        time.sleep(3)
        video_file = genai_client.files.get(name=video_file.name)
    if video_file.state.name == "FAILED":
        raise RuntimeError(f"Video processing failed for {video_path}")
    return video_file


def run_two_stage_analysis(genai_client, video_file):
    video_part = types.Part(
        file_data=types.FileData(file_uri=video_file.uri, mime_type=video_file.mime_type),
        video_metadata=types.VideoMetadata(fps=VIDEO_FPS),
    )
    stage1 = genai_client.models.generate_content(model=MODEL, contents=[video_part, STAGE1_PROMPT])
    description = stage1.text

    stage2 = genai_client.models.generate_content(
        model=MODEL,
        contents=[STAGE2_PROMPT_TEMPLATE.format(description=description)],
    )
    classification_text = stage2.text
    return description, classification_text


def parse_classification(classification_text: str):
    classification_raw = "Insufficient Information"
    confidence = "Low"
    for line in classification_text.splitlines():
        cleaned = line.strip("*- ").strip()
        lowered = cleaned.lower()
        if lowered.startswith("classification:"):
            classification_raw = cleaned.split(":", 1)[1].strip()
        elif lowered.startswith("confidence:"):
            confidence = cleaned.split(":", 1)[1].strip().rstrip(".")
    return normalize_classification(classification_raw), confidence


def ingest_clip(genai_client, embedder, collection, video_path: str, display_filename: str = None):
    """Run the full pipeline on a single video file and store the result in Chroma.
    display_filename lets the server pass the user's original upload name, since
    the file on disk may be a temp path."""
    display_filename = display_filename or os.path.basename(video_path)

    video_file = upload_and_wait(genai_client, video_path)
    description, classification_text = run_two_stage_analysis(genai_client, video_file)
    classification, confidence = parse_classification(classification_text)
    has_hedge = contains_hedge_language(description)

    embedding = embedder.encode(description).tolist()
    clip_id = str(uuid.uuid4())

    metadata = {
        "clip_id": clip_id,
        "source_filename": display_filename,
        "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
        "classification": classification,
        "confidence": confidence,
        "has_hedge": has_hedge,
        "full_classification_reasoning": classification_text,
    }

    collection.add(
        ids=[clip_id],
        embeddings=[embedding],
        documents=[description],
        metadatas=[metadata],
    )

    genai_client.files.delete(name=video_file.name)

    return {**metadata, "description": description}


def format_context(results):
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    blocks = []
    for doc, meta in zip(docs, metas):
        blocks.append(
            f"Event: {meta['source_filename']} (ingested {meta['ingestion_timestamp']})\n"
            f"Classification: {meta['classification']} (confidence: {meta['confidence']}, "
            f"contains uncertain language: {meta['has_hedge']})\n"
            f"Description: {doc}"
        )
    return "\n\n".join(blocks), metas


def query_events(embedder, collection, question: str, suspicious_only: bool = False):
    query_embedding = embedder.encode(question).tolist()
    where_filter = {"classification": "Suspicious"} if suspicious_only else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        where=where_filter,
    )

    if not results["documents"][0]:
        return {
            "answer": (
                "No events are currently indexed.\n\n"
                "This usually means the server has just started or the event database is empty. "
                "Please upload and analyze a video first, then try your question again."
            ),
            "sources": [],
        }

    context, metas = format_context(results)

    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "user",
                "content": SYNTHESIS_PROMPT_TEMPLATE.format(
                    context=context,
                    question=question,
                ),
            }
        ],
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": metas,
    }