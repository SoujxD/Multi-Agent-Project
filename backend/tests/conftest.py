"""Shared pytest configuration for the backend test suite.

Points the app at a throwaway SQLite file (not backend/local.db) so tests never
touch dev data, and forces the offline mock LLM path (mirrors the root
tests/conftest.py) so the suite is deterministic and runs without secrets or a
local Ollama server.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY"):
    os.environ.pop(key, None)
os.environ["LLM_PROVIDER"] = "mock"

_TEST_DB_PATH = Path(tempfile.gettempdir()) / "multi_agent_backend_test.db"
_TEST_DB_PATH.unlink(missing_ok=True)
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB_PATH}"
os.environ["CELERY_TASK_ALWAYS_EAGER"] = "1"


@pytest.fixture(scope="session")
def client():
    """A FastAPI TestClient wired to the throwaway SQLite DB, tables created."""
    from fastapi.testclient import TestClient

    from backend.app.db.session import init_db
    from backend.app.main import app

    init_db()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def db_session():
    """A raw SQLAlchemy session over the same throwaway test DB."""
    from backend.app.db.session import SessionLocal, init_db

    init_db()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
