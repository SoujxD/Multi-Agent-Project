"""Tests for POST /api/chat and GET /api/history."""

from __future__ import annotations


def test_chat_returns_expected_schema(client):
    response = client.post(
        "/api/chat",
        json={"user_id": "tester", "question": "Which traffic sources drive revenue?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"answer", "route", "sources", "confidence", "latency_ms"}
    assert isinstance(body["answer"], str) and body["answer"]
    assert body["route"] == "analyst"
    assert isinstance(body["sources"], list)
    assert isinstance(body["latency_ms"], int) and body["latency_ms"] >= 0


def test_chat_missing_user_id_is_rejected(client):
    response = client.post("/api/chat", json={"question": "test"})
    assert response.status_code == 422


def test_history_contains_recorded_chat(client):
    client.post(
        "/api/chat",
        json={"user_id": "history-user", "question": "Which months convert best?"},
    )
    response = client.get("/api/history", params={"user_id": "history-user"})
    assert response.status_code == 200
    entries = response.json()
    assert len(entries) >= 1
    assert entries[0]["user_id"] == "history-user"
    assert entries[0]["question"] == "Which months convert best?"


def test_history_filters_by_user_id(client):
    client.post("/api/chat", json={"user_id": "user-a", "question": "test A"})
    response = client.get("/api/history", params={"user_id": "user-does-not-exist"})
    assert response.status_code == 200
    assert response.json() == []
