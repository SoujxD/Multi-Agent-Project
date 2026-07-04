# Evaluation Dashboard

How the multi-agent analyst is benchmarked — what models are tested, which
prompt styles are compared, and which metrics decide "good."

## Models tested

`utils/llm_provider.get_chat_model()` picks the first available provider, in
this order, and is shared by the analyst, the LangGraph supervisor, and the
Ragas judge:

- **GPT** — OpenAI GPT family via `ChatOpenAI`. Set `OPENAI_API_KEY` (and
  optionally `OPENAI_CHAT_MODEL`, default `gpt-4o-mini`).
- **Groq** — fast-inference open-model host (Llama / Mixtral) via `ChatGroq`.
  Set `GROQ_API_KEY` (and optionally `GROQ_MODEL`, default
  `llama-3.3-70b-versatile`).
- **Ollama** — local open-model runtime via `ChatOllama`, for private /
  offline benchmarking. No API key: run `ollama serve` and `ollama pull
  <model>` locally, then optionally set `OLLAMA_MODEL` (default `llama3.1`)
  and `OLLAMA_BASE_URL` (default `http://localhost:11434`). Detected by a
  short reachability probe, so it's skipped instantly when not running.
- **Fallback model** — deterministic mock generator (`utils/llm_client.py`).
  Runs when no provider is configured/reachable so the whole pipeline stays
  reproducible offline.

Set `LLM_PROVIDER=openai|groq|ollama|mock` to force one provider (falls back
to mock if that provider isn't actually configured/reachable — it never
raises). Note: **tool-calling reliability varies by model.** OpenAI's
structured output reliably fills every `AnalystAnswer` field; smaller local
models (tested with Ollama's `mistral:latest`, the only locally pulled model
with tool-calling support) are less consistent, which is why the analyst
prompt sent to structured-output calls omits the legacy raw-JSON-schema
instructions the mock parser needs — that competing instruction was observed
to make Mistral dump raw JSON into a single field instead of populating
`key_findings` / `recommendations` properly.

## Prompt styles

- **Concise** — short, decision-ready answer with no filler.
- **Analytical** — structured breakdown with insights and patterns; the default for evaluation runs.
- **Source-grounded** — every claim must tie back to a retrieved chunk; hedges when evidence is weak.
- **Executive-summary** — stakeholder briefing with headline, key findings, and next steps.

## Metrics

- **Retrieval precision** — of the top-*k* retrieved chunks, how many were actually relevant to the question.
- **Faithfulness** — every factual claim in the answer must be supported by the retrieved context (no hallucination).
- **Answer relevancy** — does the answer actually address the user's question, on-topic and complete?
- **Latency** — wall-clock time from request in to response out, in milliseconds. Recorded on every `answers` row.

## How to run

```bash
# Ragas: faithfulness + answer relevancy + context precision + context recall
python main.py ragas --limit 20

# Factorial benchmark: models x prompt styles x RAG on/off, scored across
# keyword coverage, groundedness, business specificity, and JSON validity
python main.py evaluate --limit 20 --no-judge
```

Or, via the async backend:

```bash
curl -X POST http://localhost:8001/api/evaluate \
     -H "Content-Type: application/json" \
     -d '{"user_id":"me","mode":"ragas","limit":20}'
```

Results land in `outputs/` (CSV + summary JSON) and — for the backend path — in
the `evaluation_runs` table with the metric summary and `latency_ms`.
