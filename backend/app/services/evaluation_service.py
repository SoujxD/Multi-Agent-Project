"""Evaluation work function; called by the Celery task.

The ``mode="benchmark"`` branch is what the spec called "batch question testing"
- it runs the analyst over the question bank and scores every answer.
"""

from __future__ import annotations

from ..config import DATASET_PATH, OUTPUT_DIR, QUESTIONS_PATH
from ..db import crud
from ..db.session import SessionLocal
from ..utils.logging import get_logger

log = get_logger(__name__)


def run_evaluation(run_id: str, mode: str, row_limit: int, enable_judge: bool) -> None:
    """Execute a Ragas or benchmark evaluation run and update the DB row."""
    db = SessionLocal()
    try:
        crud.update_evaluation_run(db, run_id, status="running")
        if mode == "ragas":
            from evaluation.ragas_eval import run_ragas_evaluation

            summary = run_ragas_evaluation(limit=row_limit)
        else:
            from agents.analyst_agent import AnalystRAGAgent
            from evaluation.evaluator import EvaluationPipeline

            agent = AnalystRAGAgent(dataset_path=DATASET_PATH)
            pipeline = EvaluationPipeline(
                agent=agent, questions_path=QUESTIONS_PATH, output_dir=OUTPUT_DIR
            )
            df = pipeline.run(limit=row_limit, enable_judge=enable_judge)
            summary = {
                "status": "completed",
                "rows": int(len(df)),
                "columns": list(df.columns),
            }
        crud.update_evaluation_run(db, run_id, status="completed", summary=summary)
    except Exception as exc:
        log.exception("evaluation run %s failed", run_id)
        crud.update_evaluation_run(db, run_id, status="failed", error=str(exc))
    finally:
        db.close()
