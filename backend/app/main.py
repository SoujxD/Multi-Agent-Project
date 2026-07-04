"""FastAPI application entrypoint.

Run with:
    uvicorn backend.app.main:app --reload --port 8001
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config  # noqa: F401  # side-effect: puts repo root on sys.path
from .api import chat as chat_api
from .api import evaluations as evaluations_api
from .api import ingestion as ingestion_api
from .api import jobs as jobs_api
from .api import presentations as presentations_api
from .db.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Multi-Agent Analytics Backend", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


app.include_router(chat_api.router)
app.include_router(jobs_api.router)
app.include_router(presentations_api.router)
app.include_router(evaluations_api.router)
app.include_router(ingestion_api.router)
