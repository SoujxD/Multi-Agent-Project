"""CRUD helpers backed by SQLAlchemy."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import models


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(dt: datetime | None) -> datetime | None:
    """SQLite drops tzinfo on read; treat naive datetimes as UTC."""
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _latency_ms(start: datetime | None, end: datetime | None) -> int | None:
    start, end = _as_utc(start), _as_utc(end)
    if start is None or end is None:
        return None
    return int((end - start).total_seconds() * 1000)


# ─── Users / datasets ────────────────────────────────
def get_or_create_user(db: Session, user_id: str) -> models.User:
    user = db.get(models.User, user_id)
    if user is None:
        user = models.User(id=user_id)
        db.add(user)
        db.flush()
    return user


def get_or_create_dataset(db: Session, dataset_id: str) -> models.Dataset:
    dataset = db.get(models.Dataset, dataset_id)
    if dataset is None:
        dataset = models.Dataset(id=dataset_id, name=dataset_id)
        db.add(dataset)
        db.flush()
    return dataset


# ─── Chat: question + answer + retrieved chunks ─────
def record_chat(
    db: Session,
    *,
    user_id: str,
    question_text: str,
    dataset_id: str | None,
    answer_text: str,
    agent_route: str,
    retrieved_sources: list[str],
    retrieved_details: list[dict[str, Any]],
    model_name: str,
    latency_ms: int,
    confidence: float | None,
    token_count: int | None = None,
    estimated_cost: float | None = None,
) -> models.Answer:
    """Persist a full chat interaction and return the Answer row."""
    get_or_create_user(db, user_id)
    if dataset_id:
        get_or_create_dataset(db, dataset_id)

    question = models.Question(user_id=user_id, dataset_id=dataset_id, text=question_text)
    db.add(question)
    db.flush()

    answer = models.Answer(
        question_id=question.id,
        answer=answer_text,
        agent_route=agent_route,
        retrieved_sources=retrieved_sources,
        model_name=model_name,
        latency_ms=latency_ms,
        token_count=token_count,
        estimated_cost=estimated_cost,
        confidence=confidence,
    )
    db.add(answer)
    db.flush()

    for detail in retrieved_details:
        db.add(
            models.RetrievedChunk(
                answer_id=answer.id,
                rank=int(detail.get("rank", 0)),
                content=str(detail.get("content", "")),
                score=detail.get("score"),
                chunk_metadata=detail.get("metadata"),
            )
        )

    db.commit()
    db.refresh(answer)
    return answer


def list_chats(db: Session, user_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    stmt = (
        select(models.Answer, models.Question)
        .join(models.Question, models.Answer.question_id == models.Question.id)
        .order_by(models.Answer.created_at.desc())
        .limit(limit)
    )
    if user_id is not None:
        stmt = stmt.where(models.Question.user_id == user_id)

    return [
        {
            "id": answer.id,
            "user_id": question.user_id,
            "question": question.text,
            "dataset_id": question.dataset_id,
            "answer": answer.answer,
            "route": answer.agent_route,
            "sources": answer.retrieved_sources or [],
            "confidence": answer.confidence,
            "latency_ms": answer.latency_ms,
            "created_at": answer.created_at.isoformat() if answer.created_at else "",
        }
        for answer, question in db.execute(stmt).all()
    ]


# ─── Presentation jobs ───────────────────────────────
def create_presentation_job(
    db: Session, *, user_id: str, dataset_id: str | None
) -> models.PresentationJob:
    get_or_create_user(db, user_id)
    if dataset_id:
        get_or_create_dataset(db, dataset_id)
    job = models.PresentationJob(user_id=user_id, dataset_id=dataset_id, status="queued")
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def update_presentation_job(db: Session, job_id: str, **fields: Any) -> models.PresentationJob | None:
    job = db.get(models.PresentationJob, job_id)
    if job is None:
        return None
    for key, value in fields.items():
        setattr(job, key, value)
    if fields.get("status") in {"completed", "failed"} and job.completed_at is None:
        job.completed_at = _utc_now()
        job.latency_ms = _latency_ms(job.created_at, job.completed_at)
    db.commit()
    db.refresh(job)
    return job


# ─── Evaluation runs ─────────────────────────────────
def create_evaluation_run(
    db: Session,
    *,
    user_id: str,
    dataset_id: str | None,
    mode: str,
    row_limit: int,
    enable_judge: bool,
) -> models.EvaluationRun:
    get_or_create_user(db, user_id)
    if dataset_id:
        get_or_create_dataset(db, dataset_id)
    run = models.EvaluationRun(
        user_id=user_id,
        dataset_id=dataset_id,
        mode=mode,
        row_limit=row_limit,
        enable_judge=enable_judge,
        status="queued",
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def update_evaluation_run(db: Session, run_id: str, **fields: Any) -> models.EvaluationRun | None:
    run = db.get(models.EvaluationRun, run_id)
    if run is None:
        return None
    for key, value in fields.items():
        setattr(run, key, value)
    if fields.get("status") in {"completed", "failed"} and run.completed_at is None:
        run.completed_at = _utc_now()
        run.latency_ms = _latency_ms(run.created_at, run.completed_at)
    db.commit()
    db.refresh(run)
    return run


def list_evaluation_runs(
    db: Session, user_id: str | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    stmt = select(models.EvaluationRun).order_by(models.EvaluationRun.created_at.desc()).limit(limit)
    if user_id is not None:
        stmt = stmt.where(models.EvaluationRun.user_id == user_id)
    return [
        {
            "run_id": run.id,
            "user_id": run.user_id,
            "mode": run.mode,
            "limit": run.row_limit,
            "dataset_id": run.dataset_id,
            "summary": run.summary or {},
            "latency_ms": run.latency_ms,
            "created_at": run.created_at.isoformat() if run.created_at else "",
        }
        for run in db.execute(stmt).scalars().all()
    ]


# ─── Ingestion jobs ──────────────────────────────────
def create_ingestion_job(
    db: Session, *, user_id: str, dataset_id: str | None, source_length: int
) -> models.IngestionJob:
    get_or_create_user(db, user_id)
    if dataset_id:
        get_or_create_dataset(db, dataset_id)
    job = models.IngestionJob(
        user_id=user_id,
        dataset_id=dataset_id,
        status="queued",
        source_length=source_length,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def update_ingestion_job(db: Session, job_id: str, **fields: Any) -> models.IngestionJob | None:
    job = db.get(models.IngestionJob, job_id)
    if job is None:
        return None
    for key, value in fields.items():
        setattr(job, key, value)
    if fields.get("status") in {"completed", "failed"} and job.completed_at is None:
        job.completed_at = _utc_now()
        job.latency_ms = _latency_ms(job.created_at, job.completed_at)
    db.commit()
    db.refresh(job)
    return job


# ─── Unified job lookup for GET /api/jobs/{id} ──────
def find_job_any(db: Session, job_id: str) -> dict[str, Any] | None:
    """Look up a job by id in both presentation_jobs and evaluation_runs."""
    presentation = db.get(models.PresentationJob, job_id)
    if presentation is not None:
        return {
            "id": presentation.id,
            "type": "presentation",
            "status": presentation.status,
            "user_id": presentation.user_id,
            "dataset_id": presentation.dataset_id,
            "created_at": presentation.created_at.isoformat() if presentation.created_at else "",
            "completed_at": presentation.completed_at.isoformat() if presentation.completed_at else None,
            "latency_ms": presentation.latency_ms,
            "result": (
                {
                    "path": presentation.output_path,
                    "slide_count": presentation.slide_count,
                    "chart_count": presentation.chart_count,
                }
                if presentation.status == "completed"
                else None
            ),
            "error": presentation.error,
        }
    evaluation = db.get(models.EvaluationRun, job_id)
    if evaluation is not None:
        return {
            "id": evaluation.id,
            "type": "evaluate",
            "status": evaluation.status,
            "user_id": evaluation.user_id,
            "dataset_id": evaluation.dataset_id,
            "created_at": evaluation.created_at.isoformat() if evaluation.created_at else "",
            "completed_at": evaluation.completed_at.isoformat() if evaluation.completed_at else None,
            "latency_ms": evaluation.latency_ms,
            "result": (
                {"mode": evaluation.mode, "summary": evaluation.summary or {}}
                if evaluation.status == "completed"
                else None
            ),
            "error": evaluation.error,
        }
    ingestion = db.get(models.IngestionJob, job_id)
    if ingestion is not None:
        return {
            "id": ingestion.id,
            "type": "ingest",
            "status": ingestion.status,
            "user_id": ingestion.user_id,
            "dataset_id": ingestion.dataset_id,
            "created_at": ingestion.created_at.isoformat() if ingestion.created_at else "",
            "completed_at": ingestion.completed_at.isoformat() if ingestion.completed_at else None,
            "latency_ms": ingestion.latency_ms,
            "result": (
                {"chunks_added": ingestion.chunks_added, "source_length": ingestion.source_length}
                if ingestion.status == "completed"
                else None
            ),
            "error": ingestion.error,
        }
    return None
