"""Evaluation endpoints: enqueue a Celery task, list past runs.

The ``mode="benchmark"`` request is Phase 1C's "batch question testing" — it
runs the analyst over the question bank and scores every answer.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..db import crud
from ..db.session import get_db
from ..schemas.evaluation import EvaluateRequest, EvaluationRunSummary
from ..schemas.job import Job
from ..utils.logging import get_logger
from ..workers.worker import run_evaluation_task

router = APIRouter(prefix="/api", tags=["evaluation"])
log = get_logger(__name__)


@router.post("/evaluate", response_model=Job)
def post_evaluate(payload: EvaluateRequest, db: Session = Depends(get_db)) -> dict:
    """Create an evaluation run (status='queued') and dispatch to Celery."""
    run = crud.create_evaluation_run(
        db,
        user_id=payload.user_id,
        dataset_id=payload.dataset_id,
        mode=payload.mode,
        row_limit=payload.limit,
        enable_judge=payload.enable_judge,
    )
    run_evaluation_task.delay(run.id, payload.mode, payload.limit, payload.enable_judge)
    return {
        "id": run.id,
        "type": "evaluate",
        "status": run.status,  # "queued"
        "user_id": run.user_id,
        "dataset_id": run.dataset_id,
        "created_at": run.created_at.isoformat() if run.created_at else "",
        "completed_at": None,
        "latency_ms": None,
        "result": None,
        "error": None,
    }


@router.get("/evaluation-runs", response_model=list[EvaluationRunSummary])
def get_evaluation_runs(
    user_id: str | None = None,
    limit: int = 50,
    db: Session = Depends(get_db),
) -> list[dict]:
    """Return evaluation-run summaries, most recent first."""
    return crud.list_evaluation_runs(db, user_id=user_id, limit=limit)
