"""Tests for retrieval: chat returns retrieved sources, and ingestion adds
new content that becomes retrievable.

The ingestion test monkeypatches the Chroma persist path to a temp directory
so it never writes into the real project's shared ``data/chroma_db``.
"""

from __future__ import annotations


def test_chat_with_rag_returns_retrieved_sources(client):
    response = client.post(
        "/api/chat",
        json={"user_id": "tester", "question": "Which traffic sources drive revenue?"},
    )
    assert response.status_code == 200
    sources = response.json()["sources"]
    assert len(sources) > 0
    assert all(isinstance(source, str) and source for source in sources)


def test_ingest_document_creates_completed_job_with_chunks(client, monkeypatch, tmp_path):
    from backend.app.services import ingestion_service

    monkeypatch.setattr(ingestion_service, "REPO_ROOT", tmp_path)

    document = (
        "Q3 revenue rose sharply, driven by returning-visitor traffic on mobile devices. "
        "The New_Visitor cohort still lags on conversion, but engagement grew significantly. "
        "Bounce rates dropped once the homepage carousel was replaced with a personalized grid."
    )
    response = client.post(
        "/api/ingest",
        json={
            "user_id": "tester",
            "dataset_id": "test-notes",
            "text": document,
            "chunk_size": 100,
            "chunk_overlap": 20,
        },
    )
    assert response.status_code == 200
    job = response.json()
    assert job["type"] == "ingest"
    assert job["status"] == "queued"

    final = client.get(f"/api/jobs/{job['id']}").json()
    assert final["status"] == "completed"
    assert final["result"]["chunks_added"] > 0
    assert final["result"]["source_length"] == len(document)
    assert (tmp_path / "data" / "chroma_db").exists()
