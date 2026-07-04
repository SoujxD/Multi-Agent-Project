"""Celery worker: dispatches async jobs for deck generation, RAGAS evaluation,
batch question testing (via ``evaluation_service`` in ``benchmark`` mode), and
document ingestion.

Run the worker (production):

    celery -A backend.app.workers.worker.celery_app worker --loglevel=info --pool=solo

Environment variables:

- ``CELERY_BROKER_URL``  - defaults to ``redis://localhost:6379/0``. Use a
  ``amqp://`` URL to talk to RabbitMQ instead.
- ``CELERY_RESULT_BACKEND`` - defaults unset; the Postgres/SQLite ``jobs`` tables
  are the source of truth.
- ``CELERY_TASK_ALWAYS_EAGER=1`` - forces synchronous, in-process execution
  (no broker required). Defaults to eager when ``CELERY_BROKER_URL`` is unset.
"""

from __future__ import annotations

import os

from celery import Celery


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


BROKER_URL = os.getenv("CELERY_BROKER_URL")
BROKER_CONFIGURED = bool(BROKER_URL)
BROKER_URL = BROKER_URL or "memory://"  # in-memory broker used only in eager mode
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND")  # optional; DB is source of truth

# Default to eager when no broker is configured, so the app runs on plain dev
# machines without Redis or RabbitMQ. Override with CELERY_TASK_ALWAYS_EAGER=0.
_env_eager = os.getenv("CELERY_TASK_ALWAYS_EAGER")
TASK_ALWAYS_EAGER = _truthy(_env_eager) if _env_eager is not None else not BROKER_CONFIGURED

celery_app = Celery("multi_agent_backend", broker=BROKER_URL, backend=RESULT_BACKEND)
celery_app.conf.update(
    task_always_eager=TASK_ALWAYS_EAGER,
    task_eager_propagates=True,
    broker_connection_retry_on_startup=False,
    task_ignore_result=True,
    worker_hijack_root_logger=False,
)


# ─── Tasks ─────────────────────────────────────────────────────────
@celery_app.task(name="worker.generate_deck")
def generate_deck_task(job_id: str) -> None:
    from ..services.presentation_service import run_presentation

    run_presentation(job_id)


@celery_app.task(name="worker.run_evaluation")
def run_evaluation_task(run_id: str, mode: str, row_limit: int, enable_judge: bool) -> None:
    from ..services.evaluation_service import run_evaluation

    run_evaluation(run_id, mode, row_limit, enable_judge)


@celery_app.task(name="worker.ingest_document")
def ingest_document_task(
    job_id: str, text: str, dataset_id: str | None, chunk_size: int, chunk_overlap: int
) -> None:
    from ..services.ingestion_service import run_ingestion

    run_ingestion(job_id, text, dataset_id, chunk_size, chunk_overlap)
