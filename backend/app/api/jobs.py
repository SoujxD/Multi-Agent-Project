"""Job polling endpoint. Looks up in both presentation_jobs and evaluation_runs."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import crud
from ..db.session import get_db
from ..schemas.job import Job

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=Job)
def get_job(job_id: str, db: Session = Depends(get_db)) -> dict:
    job = crud.find_job_any(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    return job
