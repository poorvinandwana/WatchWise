from fastapi import FastAPI
from app.config import *

app = FastAPI()

@app.get("/")
def home():
    return {"status": "ok"}