"""Chat-model provider selection: OpenAI -> Groq -> Ollama -> None (mock).

Centralizes the "which real LLM do we call" decision so the analyst
(``agents/analyst_agent.py``), the LangGraph supervisor (``agents/graph.py``),
and the Ragas judge (``evaluation/ragas_eval.py``) all pick the same provider
the same way, and so callers can record which model actually answered instead
of guessing from ``OPENAI_API_KEY`` alone.

Priority order (first available wins): **OpenAI -> Groq -> Ollama -> mock**.
Set ``LLM_PROVIDER`` to force one provider; if it isn't configured/reachable,
``get_chat_model`` returns ``None`` rather than raising, so callers fall back
to the deterministic mock.

Environment variables:

- ``OPENAI_API_KEY`` / ``OPENAI_CHAT_MODEL`` (default ``gpt-4o-mini``)
- ``GROQ_API_KEY`` / ``GROQ_MODEL`` (default ``llama-3.3-70b-versatile``)
- ``OLLAMA_BASE_URL`` (default ``http://localhost:11434``) / ``OLLAMA_MODEL``
  (default ``llama3.1``) - no API key needed, but the server must be running
  locally (``ollama serve`` + ``ollama pull <model>``).
- ``LLM_PROVIDER`` - force ``"openai"``, ``"groq"``, ``"ollama"``, or
  ``"mock"`` (the last always returns ``None``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_OLLAMA_MODEL = "llama3.1"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

_PROVIDER_ORDER = ("openai", "groq", "ollama")


@dataclass(slots=True)
class ChatModelChoice:
    """A resolved chat model plus the name to record for observability."""

    llm: Any
    provider: str
    model_name: str


def _ollama_reachable(base_url: str, timeout: float = 0.3) -> bool:
    """Cheap local reachability probe so we don't hang on a server that isn't running."""
    try:
        import urllib.request

        urllib.request.urlopen(f"{base_url}/api/tags", timeout=timeout)
        return True
    except Exception:
        return False


def get_chat_model(temperature: float = 0.0) -> ChatModelChoice | None:
    """Return the first available real chat model, or ``None`` to signal the mock fallback."""
    forced = os.getenv("LLM_PROVIDER", "").strip().lower()
    if forced == "mock":
        return None
    providers = (forced,) if forced in _PROVIDER_ORDER else _PROVIDER_ORDER

    for provider in providers:
        if provider == "openai" and os.getenv("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI

            model_name = os.getenv("OPENAI_CHAT_MODEL", DEFAULT_OPENAI_MODEL)
            return ChatModelChoice(
                llm=ChatOpenAI(model=model_name, temperature=temperature),
                provider="openai",
                model_name=model_name,
            )

        if provider == "groq" and os.getenv("GROQ_API_KEY"):
            from langchain_groq import ChatGroq

            model_name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
            return ChatModelChoice(
                llm=ChatGroq(model=model_name, temperature=temperature),
                provider="groq",
                model_name=model_name,
            )

        if provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
            if _ollama_reachable(base_url):
                from langchain_ollama import ChatOllama

                model_name = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
                return ChatModelChoice(
                    llm=ChatOllama(model=model_name, base_url=base_url, temperature=temperature),
                    provider="ollama",
                    model_name=model_name,
                )

    return None
