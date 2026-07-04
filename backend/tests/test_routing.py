"""Tests for job creation/status routing: presentation jobs, and the unified
``GET /api/jobs/{job_id}`` lookup discriminating between job tables.
"""

from __future__ import annotations


def test_presentation_job_completes_with_deck_metadata(client):
    response = client.post("/api/presentation", json={"user_id": "tester"})
    assert response.status_code == 200
    job = response.json()
    assert job["type"] == "presentation"
    assert job["status"] == "queued"

    final = client.get(f"/api/jobs/{job['id']}").json()
    assert final["status"] == "completed"
    assert final["result"]["slide_count"] > 0
    assert final["result"]["chart_count"] > 0
    assert final["result"]["path"].endswith(".pptx")


def test_unknown_job_id_returns_404(client):
    response = client.get("/api/jobs/does-not-exist")
    assert response.status_code == 404


def test_job_lookup_routes_to_correct_type(client):
    """A presentation job and an evaluation run must resolve to distinct types."""
    presentation_job = client.post("/api/presentation", json={"user_id": "tester"}).json()
    evaluation_job = client.post(
        "/api/evaluate", json={"user_id": "tester", "mode": "ragas", "limit": 1}
    ).json()

    presentation_lookup = client.get(f"/api/jobs/{presentation_job['id']}").json()
    evaluation_lookup = client.get(f"/api/jobs/{evaluation_job['id']}").json()

    assert presentation_lookup["type"] == "presentation"
    assert evaluation_lookup["type"] == "evaluate"
    assert presentation_lookup["id"] != evaluation_lookup["id"]
