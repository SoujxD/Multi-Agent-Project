"""Job schema shared by background-task endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


# NOTE: "pending" retained for backwards compatibility; new jobs enter as "queued"
# because the Celery task is enqueued to the broker before the worker picks it up.
JobStatus = Literal["queued", "running", "completed", "failed", "pending"]
JobType = Literal["chat", "presentation", "evaluate", "ingest"]


class Job(BaseModel):
    id: str
    type: JobType
    status: JobStatus
    user_id: str | None = None
    dataset_id: str | None = None
    created_at: str
    completed_at: str | None = None
    latency_ms: int | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
