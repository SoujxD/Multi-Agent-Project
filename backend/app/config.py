"""Backend settings and shared paths.

Also puts the repo root on ``sys.path`` so the backend routers can reuse the
existing top-level ``agents/``, ``evaluation/``, and ``utils/`` packages.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"
DATASET_PATH = DATA_DIR / "dataset.csv"
QUESTIONS_PATH = DATA_DIR / "evaluation_questions.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# DB URL exposed for logs / debugging. The engine is built in db.session.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///backend/local.db")
