"""Schemas for the evaluation endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class EvaluateRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    mode: Literal["ragas", "benchmark"] = "ragas"
    limit: int = Field(default=20, ge=1, le=100)
    enable_judge: bool = False
    dataset_id: str | None = None


class EvaluationRunSummary(BaseModel):
    run_id: str
    user_id: str
    mode: str
    limit: int
    dataset_id: str | None = None
    summary: dict[str, Any]
    latency_ms: int | None = None
    created_at: str
