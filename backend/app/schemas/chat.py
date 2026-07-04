"""Request/response schemas for the chat endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    dataset_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    route: str
    sources: list[str]
    confidence: float | None
    latency_ms: int


class ChatHistoryEntry(BaseModel):
    """A stored chat interaction, returned by ``GET /api/history``."""

    id: str
    user_id: str
    question: str
    dataset_id: str | None = None
    answer: str
    route: str
    sources: list[str]
    confidence: float | None
    latency_ms: int
    created_at: str
