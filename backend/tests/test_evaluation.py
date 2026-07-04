"""Tests for POST /api/evaluate and GET /api/evaluation-runs.

Celery runs in eager mode during tests (see conftest.py), so by the time the
POST handler returns, the task has already executed synchronously and updated
the DB row -- but the response body reflects the row's state *before* dispatch
(status="queued"), matching real async behavior. The follow-up GET reflects
the true final state.
"""

from __future__ import annotations


def test_evaluate_ragas_queues_then_completes(client):
    response = client.post(
        "/api/evaluate",
        json={"user_id": "tester", "mode": "ragas", "limit": 2},
    )
    assert response.status_code == 200
    job = response.json()
    assert job["type"] == "evaluate"
    assert job["status"] == "queued"

    followup = client.get(f"/api/jobs/{job['id']}")
    assert followup.status_code == 200
    final = followup.json()
    assert final["status"] == "completed"
    assert final["result"]["mode"] == "ragas"
    # No LLM judge available in tests -> graceful skip, not a crash.
    assert final["result"]["summary"]["status"] == "skipped"


def test_evaluate_benchmark_scores_rows(client):
    response = client.post(
        "/api/evaluate",
        json={"user_id": "tester", "mode": "benchmark", "limit": 1, "enable_judge": False},
    )
    job_id = response.json()["id"]

    final = client.get(f"/api/jobs/{job_id}").json()
    assert final["status"] == "completed"
    assert final["result"]["summary"]["status"] == "completed"
    assert final["result"]["summary"]["rows"] > 0


def test_evaluation_runs_lists_recorded_runs(client):
    client.post("/api/evaluate", json={"user_id": "runs-user", "mode": "ragas", "limit": 1})
    response = client.get("/api/evaluation-runs", params={"user_id": "runs-user"})
    assert response.status_code == 200
    runs = response.json()
    assert len(runs) >= 1
    assert runs[0]["user_id"] == "runs-user"
    assert runs[0]["mode"] == "ragas"
