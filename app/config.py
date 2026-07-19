from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

CHROMA_DB_PATH = os.getenv(
    "CHROMA_DB_PATH",
    str(DATA_DIR / "chroma")
)


MODEL = "gemini-2.5-flash"
VIDEO_FPS = 5.0


COLLECTION_NAME = "watchwise_events"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"

TOP_K = 5

HEDGE_WORDS = [
    "possibly",
    "appears to",
    "likely",
    "not clearly",
    "unclear",
    "may be",
    "seem",
    "resembling",
    "no longer clearly visible",
]

Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)