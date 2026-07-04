"""Schemas for the presentation endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PresentationRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    dataset_id: str | None = None
