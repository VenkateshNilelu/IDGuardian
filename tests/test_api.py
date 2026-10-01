"""
Tests for the non-scoring API surface: /healthz and the History/Saved Reports
routes.

History/Saved tests use a clearly-marked test client_id and clean up after
themselves (DELETE at the end) since -- when MONGODB_URI is configured --
these routes write to the real Atlas cluster, not a mock.
"""
import uuid


def test_healthz_reports_models_loaded(client):
    resp = client.get("/healthz")
    assert resp.status_code in (200, 503)
    data = resp.get_json()
    assert "status" in data
    assert "models_loaded" in data
    assert data["models_loaded"] is True, "Models failed to load at startup"
    assert data["demo_profiles_loaded"] > 0


def _sample_result(app_module, platform="instagram"):
    username = app_module.demo_profiles[platform]["username"].iloc[0]
    return {
        "platform": platform,
        "username": username,
        "ground_truth_archetype": "Test",
        "ground_truth_is_fake": False,
        "fusion_score": 0.12,
    }


def test_history_round_trip(app_module, client):
    if not app_module.db.DB_AVAILABLE:
        import pytest
        pytest.skip("MongoDB not configured/available -- history is localStorage-only client-side")

    test_client_id = f"pytest-{uuid.uuid4().hex[:12]}"
    result = _sample_result(app_module)

    try:
        post_resp = client.post("/api/history", json={"client_id": test_client_id, "result": result})
        assert post_resp.status_code == 200
        entry_id = post_resp.get_json()["id"]

        get_resp = client.get(f"/api/history?client_id={test_client_id}")
        assert get_resp.status_code == 200
        entries = get_resp.get_json()
        assert len(entries) == 1
        assert entries[0]["id"] == entry_id
        assert entries[0]["username"] == result["username"]

        del_resp = client.delete(f"/api/history/{entry_id}?client_id={test_client_id}")
        assert del_resp.status_code == 200

        get_resp2 = client.get(f"/api/history?client_id={test_client_id}")
        assert get_resp2.get_json() == []
    finally:
        client.delete(f"/api/history?client_id={test_client_id}")  # cleanup even on assertion failure


def test_history_requires_client_id(client):
    resp = client.get("/api/history")
    assert resp.status_code in (400, 503)  # 503 if DB unavailable, 400 if available but no client_id
