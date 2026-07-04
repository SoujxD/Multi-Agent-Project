"""Shared pytest configuration.

Ensures the repository root is importable and forces the project into its
deterministic offline mock mode during tests by clearing any API keys. This
makes the suite reproducible and lets it run in CI without secrets.
"""

import os
import sys
from pathlib import Path

# Force the non-interactive Agg backend before matplotlib is imported anywhere
# (agents/presentation_agent.py uses it to render charts). Without this, a
# machine with a display can select an interactive Tkinter backend, which
# raises "main thread is not in main loop" during interpreter teardown when
# charts are generated off the main thread (e.g. via FastAPI's TestClient).
os.environ.setdefault("MPLBACKEND", "Agg")

# Make the repo root importable (so `import agents...` / `import evaluation...` work).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Force offline mock paths: no LLM calls, deterministic behavior.
for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY"):
    os.environ.pop(key, None)

# utils.llm_provider.get_chat_model() also checks for a locally-reachable Ollama
# server, independent of any API key. A dev machine with Ollama running would
# otherwise make tests non-deterministic (and slow). Force the mock path.
os.environ["LLM_PROVIDER"] = "mock"
