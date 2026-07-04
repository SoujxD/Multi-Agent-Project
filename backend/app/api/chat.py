"""Chat endpoint: run the analyst synchronously and persist the interaction."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import crud
from ..db.session import get_db
from ..schemas.chat import ChatHistoryEntry, ChatRequest, ChatResponse
from ..utils.logging import get_logger

router = APIRouter(prefix="/api", tags=["chat"])
log = get_logger(__name__)


@router.post("/chat", response_model=ChatResponse)
def post_chat(payload: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    """Run the LangChain analyst and persist question + answer + retrieved chunks."""
    from agents.analyst_agent import run_analyst_lc

    start = time.perf_counter()
    try:
        result = run_analyst_lc(payload.question, use_rag=True)
    except Exception as exc:
        log.exception("chat failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    latency_ms = int((time.perf_counter() - start) * 1000)

    answer_text = str(result.get("answer", ""))
    sources = list(result.get("retrieved_contexts", []))
    confidence_raw = result.get("confidence")
    confidence = float(confidence_raw) if isinstance(confidence_raw, (int, float)) else None

    # TODO(review): route through the LangGraph supervisor to set this dynamically.
    route = "analyst"
    model_name = str(result.get("model_name", "mock"))
    # TODO(review): populate token_count / estimated_cost from the chat model's
    # usage_metadata when we're on a real (non-mock) provider path.
    token_count: int | None = None
    estimated_cost: float | None = None

    retrieved_details = [
        {"rank": index + 1, "content": source, "score": None, "metadata": None}
        for index, source in enumerate(sources)
    ]

    answer_row = crud.record_chat(
        db,
        user_id=payload.user_id,
        question_text=payload.question,
        dataset_id=payload.dataset_id,
        answer_text=answer_text,
        agent_route=route,
        retrieved_sources=sources,
        retrieved_details=retrieved_details,
        model_name=model_name,
        latency_ms=latency_ms,
        confidence=confidence,
        token_count=token_count,
        estimated_cost=estimated_cost,
    )

    return ChatResponse(
        answer=answer_row.answer,
        route=answer_row.agent_route,
        sources=answer_row.retrieved_sources or [],
        confidence=answer_row.confidence,
        latency_ms=answer_row.latency_ms,
    )


@router.get("/history", response_model=list[ChatHistoryEntry])
def get_history(
    user_id: str | None = None,
    limit: int = 50,
    db: Session = Depends(get_db),
) -> list[dict]:
    """Return recent chat interactions, most recent first."""
    return crud.list_chats(db, user_id=user_id, limit=limit)
