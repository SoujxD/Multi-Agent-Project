"""SQLAlchemy 2.0 ORM models for Phase 1B.

The schema is intentionally portable across SQLite (default for local dev) and
Postgres (set ``DATABASE_URL`` to a ``postgresql+psycopg://...`` URL). The
``JSON`` column type maps to ``JSONB`` on Postgres and ``TEXT`` on SQLite.

Nine tables:
- ``users``               user accounts (id supplied by the client)
- ``datasets``            dataset registry (id supplied by the client)
- ``questions``           user chat inputs
- ``answers``             AI responses with the rich metadata bundle
- ``retrieved_chunks``    the retrieved context per answer, one row per source
- ``agent_runs``          LangGraph supervisor traces
- ``presentation_jobs``   deck-generation background jobs
- ``evaluation_runs``     Ragas / benchmark evaluation batches
- ``feedback``            historical log of user feedback on answers
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid_hex() -> str:
    return uuid.uuid4().hex


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────
# Users & datasets
# ─────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    email: Mapped[str | None] = mapped_column(String(320), unique=True, nullable=True)
    name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    source_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    schema_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)


# ─────────────────────────────────────────
# Chat: questions / answers / chunks
# ─────────────────────────────────────────
class Question(Base):
    __tablename__ = "questions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid_hex)
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("users.id"), nullable=False, index=True)
    dataset_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("datasets.id"), nullable=True, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)

    answers: Mapped[list["Answer"]] = relationship(back_populates="question", cascade="all, delete-orphan")


class Answer(Base):
    """The AI response. Stores every field listed in the Phase 1B spec."""

    __tablename__ = "answers"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid_hex)
    question_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("questions.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Content
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    agent_route: Mapped[str] = mapped_column(String(64), nullable=False)
    retrieved_sources: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)

    # Model + cost
    model_name: Mapped[str] = mapped_column(String(200), nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estimated_cost: Mapped[float | None] = mapped_column(Numeric(12, 6), nullable=True)

    # Analyst confidence (Pydantic AnalystAnswer.confidence)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Evaluation scores (set later by a Ragas run)
    faithfulness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_precision: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Quick user-feedback flag (thumbs, rating, or comment summary). Full log
    # lives in the ``feedback`` table.
    user_feedback: Mapped[str | None] = mapped_column(String(200), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)

    question: Mapped[Question] = relationship(back_populates="answers")
    chunks: Mapped[list["RetrievedChunk"]] = relationship(
        back_populates="answer", cascade="all, delete-orphan"
    )
    feedback_events: Mapped[list["Feedback"]] = relationship(
        back_populates="answer", cascade="all, delete-orphan"
    )


class RetrievedChunk(Base):
    """One retrieved context row (per Answer). Enables joined analytics."""

    __tablename__ = "retrieved_chunks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid_hex)
    answer_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("answers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Renamed from ``metadata`` because that name is reserved by DeclarativeBase.
    chunk_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    answer: Mapped[Answer] = relationship(back_populates="chunks")


# ─────────────────────────────────────────
# LangGraph traces
# ─────────────────────────────────────────
class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid_hex)
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("users.id"), nullable=False, index=True)
    question_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("questions.id"), nullable=True, index=True
    )
    route: Mapped[str] = mapped_column(String(64), nullable=False)
    steps: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="completed")
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)


# ─────────────────────────────────────────
# Presentation & evaluation jobs
# ─────────────────────────────────────────
class PresentationJob(Base):
    __tablename__ = "presentation_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid_hex)
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("users.id"), nullable=False, index=True)
    dataset_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("datasets.id"), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    output_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    slide_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chart_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid_hex)
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("users.id"), nullable=False, index=True)
    dataset_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("datasets.id"), nullable=True)
    mode: Mapped[str] = mapped_column(String(32), nullable=False)  # "ragas" | "benchmark"
    # Column named ``row_limit`` (not ``limit``) because LIMIT is a SQL keyword.
    row_limit: Mapped[int] = mapped_column(Integer, nullable=False)
    enable_judge: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# ─────────────────────────────────────────
# Ingestion jobs (async worker)
# ─────────────────────────────────────────
class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid_hex)
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("users.id"), nullable=False, index=True)
    dataset_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("datasets.id"), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    source_length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunks_added: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# ─────────────────────────────────────────
# Feedback
# ─────────────────────────────────────────
class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid_hex)
    answer_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("answers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("users.id"), nullable=False, index=True)
    rating: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 1-5 or -1/+1
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)

    answer: Mapped[Answer] = relationship(back_populates="feedback_events")
