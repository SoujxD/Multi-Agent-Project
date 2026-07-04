"""Schemas for the document-ingestion endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1, description="Raw document text to chunk and index.")
    dataset_id: str | None = None
    chunk_size: int = Field(default=500, ge=64, le=4000)
    chunk_overlap: int = Field(default=100, ge=0, le=1000)
