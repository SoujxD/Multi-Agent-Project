"""Document ingestion endpoint: enqueue a Celery task to chunk + embed text."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..db import crud
from ..db.session import get_db
from ..schemas.ingestion import IngestRequest
from ..schemas.job import Job
from ..utils.logging import get_logger
from ..workers.worker import ingest_document_task

router = APIRouter(prefix="/api", tags=["ingestion"])
log = get_logger(__name__)


@router.post("/ingest", response_model=Job)
def post_ingest(payload: IngestRequest, db: Session = Depends(get_db)) -> dict:
    """Create an ingestion job and dispatch chunking + embedding to Celery."""
    job = crud.create_ingestion_job(
        db,
        user_id=payload.user_id,
        dataset_id=payload.dataset_id,
        source_length=len(payload.text),
    )
    ingest_document_task.delay(
        job.id,
        payload.text,
        payload.dataset_id,
        payload.chunk_size,
        payload.chunk_overlap,
    )
    return {
        "id": job.id,
        "type": "ingest",
        "status": job.status,  # "queued"
        "user_id": job.user_id,
        "dataset_id": job.dataset_id,
        "created_at": job.created_at.isoformat() if job.created_at else "",
        "completed_at": None,
        "latency_ms": None,
        "result": None,
        "error": None,
    }
