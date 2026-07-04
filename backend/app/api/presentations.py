"""Presentation endpoint: enqueue a Celery task, poll ``/api/jobs/{id}``."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..db import crud
from ..db.session import get_db
from ..schemas.job import Job
from ..schemas.presentation import PresentationRequest
from ..utils.logging import get_logger
from ..workers.worker import generate_deck_task

router = APIRouter(prefix="/api", tags=["presentation"])
log = get_logger(__name__)


@router.post("/presentation", response_model=Job)
def post_presentation(payload: PresentationRequest, db: Session = Depends(get_db)) -> dict:
    """Create a presentation job (status='queued') and dispatch to Celery."""
    job = crud.create_presentation_job(db, user_id=payload.user_id, dataset_id=payload.dataset_id)
    generate_deck_task.delay(job.id)
    return {
        "id": job.id,
        "type": "presentation",
        "status": job.status,  # "queued"
        "user_id": job.user_id,
        "dataset_id": job.dataset_id,
        "created_at": job.created_at.isoformat() if job.created_at else "",
        "completed_at": None,
        "latency_ms": None,
        "result": None,
        "error": None,
    }
