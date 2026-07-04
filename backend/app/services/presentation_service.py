"""Deck-generation work function; called by the Celery task."""

from __future__ import annotations

from ..config import DATASET_PATH, OUTPUT_DIR
from ..db import crud
from ..db.session import SessionLocal
from ..utils.logging import get_logger

log = get_logger(__name__)


def run_presentation(job_id: str) -> None:
    """Generate the deck and update the presentation_jobs row."""
    from agents.presentation_agent import PresentationGeneratorAgent

    db = SessionLocal()
    try:
        crud.update_presentation_job(db, job_id, status="running")
        presenter = PresentationGeneratorAgent(output_dir=OUTPUT_DIR)
        output_path, slides, chart_paths = presenter.create_presentation(
            dataset_path=DATASET_PATH,
            output_path=OUTPUT_DIR / "presentation.pptx",
        )
        crud.update_presentation_job(
            db,
            job_id,
            status="completed",
            output_path=str(output_path),
            slide_count=len(slides),
            chart_count=len(chart_paths),
        )
    except Exception as exc:
        log.exception("presentation job %s failed", job_id)
        crud.update_presentation_job(db, job_id, status="failed", error=str(exc))
    finally:
        db.close()
