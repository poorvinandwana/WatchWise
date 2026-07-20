from fastapi import FastAPI

from app.config import *
from app.core import *

app = FastAPI()

@app.get("/")
def home():
    return {"status": "ok"}