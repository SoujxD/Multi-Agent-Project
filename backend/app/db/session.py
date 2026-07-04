"""SQLAlchemy engine, session factory, and startup hook.

Defaults to a local SQLite file (`backend/local.db`) for zero-setup dev. Set
``DATABASE_URL`` to a Postgres URL (e.g.
``postgresql+psycopg://user:pass@host:5432/dbname``) for prod. The schema is
portable, but Postgres requires the ``psycopg[binary]`` package installed.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base


_DEFAULT_SQLITE = Path(__file__).resolve().parents[2] / "local.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{_DEFAULT_SQLITE}")

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, future=True, echo=False, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Create all tables. Idempotent."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Iterator[Session]:
    """FastAPI dependency: yield a scoped session for the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
