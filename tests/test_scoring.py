"""
Tests for score_profile() / the four-layer pipeline, exercised through a real
demo lookup rather than calling score_profile() directly -- that way a bug in
parse_profile_input() or the lookup route would also be caught here.

Picks a real username from the live demo dataset at test time (app_module
.demo_profiles) instead of hardcoding one, so this doesn't go stale if the
dataset is regenerated with a different set of synthetic usernames.
"""


def test_lookup_returns_well_formed_result(app_module, client):
    platform = "instagram"
    username = app_module.demo_profiles[platform]["username"].iloc[0]

    resp = client.get(f"/api/lookup?query={username}")
    assert resp.status_code == 200
    data = resp.get_json()

    assert data["username"].lower() == username.lower()
    assert data["platform"] == platform
    assert "ground_truth_archetype" in data
    assert isinstance(data["ground_truth_is_fake"], bool)

    for layer in ("lexical_score", "semantic_score", "behavioral_score", "fusion_score"):
        assert 0.0 <= data[layer] <= 1.0, f"{layer} out of [0,1]: {data[layer]}"

    assert data["verdict"] in ("Likely Fake", "Likely Genuine")
    assert data["analysis_time_seconds"] >= 0

    explanations = data["explanations"]
    for layer in ("lexical", "semantic", "behavioral", "fusion"):
        assert layer in explanations
        assert "summary" in explanations[layer]

    assert "behavioral" in data["profile_data"]
    assert len(data["profile_data"]["behavioral"]) > 0


def test_lookup_across_all_platforms(app_module, client):
    """One real lookup per platform -- catches a platform-specific regression
    (e.g. a missing behavioral column) that a single-platform test would miss."""
    for platform in app_module.T.PLATFORMS:
        username = app_module.demo_profiles[platform]["username"].iloc[0]
        resp = client.get(f"/api/lookup?query={username}")
        assert resp.status_code == 200, f"{platform} lookup failed: {resp.get_json()}"
        assert resp.get_json()["platform"] == platform


def test_lookup_unknown_username_returns_404(client):
    resp = client.get("/api/lookup?query=definitely_not_a_real_demo_username_xyz123")
    assert resp.status_code == 404
    assert "error" in resp.get_json()


def test_lookup_empty_query_returns_400(client):
    resp = client.get("/api/lookup?query=")
    assert resp.status_code == 400


def test_dotted_username_lookup_does_not_404(app_module, client):
    """Direct regression test for the dotted-username parsing bug: find a real
    demo username containing a '.' and confirm it resolves, not just that the
    parser alone returns the right tuple (test_parsing.py covers that)."""
    dotted = None
    for platform in app_module.T.PLATFORMS:
        usernames = app_module.demo_profiles[platform]["username"]
        hits = usernames[usernames.str.contains(r"\.", regex=True)]
        if len(hits):
            dotted = hits.iloc[0]
            break
    if dotted is None:
        import pytest
        pytest.skip("No dotted username in the current demo dataset to test against")

    resp = client.get(f"/api/lookup?query={dotted}")
    assert resp.status_code == 200, f"Dotted username '{dotted}' failed to resolve"
