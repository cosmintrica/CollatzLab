from __future__ import annotations

from fastapi.testclient import TestClient

from collatz_lab.api import create_app
from collatz_lab.config import Settings


def test_api_summary_and_run_flow(settings: Settings):
    app = create_app(settings)
    client = TestClient(app)

    summary = client.get("/api/summary")
    assert summary.status_code == 200
    assert summary.json()["direction_count"] == 3

    run_response = client.post(
        "/api/runs",
        json={
            "direction_slug": "verification",
            "name": "api-run",
            "range_start": 1,
            "range_end": 25,
        },
    )
    assert run_response.status_code == 200
    run_payload = run_response.json()
    assert run_payload["status"] == "completed"

    validate_response = client.post(f"/api/runs/{run_payload['id']}/validate")
    assert validate_response.status_code == 200
    assert validate_response.json()["status"] == "validated"

    claim_response = client.post(
        "/api/claims",
        json={
            "direction_slug": "lemma-workspace",
            "title": "API claim",
            "statement": "The API should persist claims.",
        },
    )
    assert claim_response.status_code == 200

    link_response = client.post(
        "/api/claims/link-run",
        json={
            "claim_id": claim_response.json()["id"],
            "run_id": run_payload["id"],
            "relation": "supports",
        },
    )
    assert link_response.status_code == 200

    links_response = client.get("/api/claim-run-links")
    assert links_response.status_code == 200
    assert any(link["run_id"] == run_payload["id"] for link in links_response.json())

    artifacts_response = client.get("/api/artifacts")
    assert artifacts_response.status_code == 200
    claim_artifact = next(
        artifact for artifact in artifacts_response.json() if artifact["claim_id"] == claim_response.json()["id"]
    )

    artifact_content = client.get(f"/api/artifacts/{claim_artifact['id']}/content")
    assert artifact_content.status_code == 200
    assert "API claim" in artifact_content.json()["content"]

    artifact_download = client.get(f"/api/artifacts/{claim_artifact['id']}/download")
    assert artifact_download.status_code == 200
    assert "attachment" in artifact_download.headers["content-disposition"]

    links_response = client.get("/api/claim-run-links")
    assert links_response.status_code == 200
    assert links_response.json()[0]["claim_id"] == claim_response.json()["id"]

    artifact_response = client.get("/api/artifacts")
    assert artifact_response.status_code == 200
    artifact_payload = next(
        artifact
        for artifact in artifact_response.json()
        if artifact["claim_id"] == claim_response.json()["id"]
    )

    artifact_content_response = client.get(
        f"/api/artifacts/{artifact_payload['id']}/content"
    )
    assert artifact_content_response.status_code == 200
    assert claim_response.json()["title"] in artifact_content_response.json()["text"]


def test_api_rejects_non_executable_run(settings: Settings):
    app = create_app(settings)
    client = TestClient(app)

    response = client.post(
        "/api/runs",
        json={
            "direction_slug": "verification",
            "name": "gpu-run",
            "range_start": 10,
            "range_end": 20,
            "kernel": "cpu-direct",
            "hardware": "gpu",
            "enqueue_only": True,
        },
    )
    assert response.status_code == 400

    hardware = client.get("/api/hardware")
    assert hardware.status_code == 200
    assert any(item["kind"] == "cpu" for item in hardware.json())


def test_api_source_registry_and_consensus_probe(settings: Settings):
    app = create_app(settings)
    client = TestClient(app)

    llm_status = client.get("/api/llm/status")
    assert llm_status.status_code == 200
    assert llm_status.json()["provider"] == "gemini"
    assert llm_status.json()["ready"] is False

    source_response = client.post(
        "/api/sources",
        json={
            "direction_slug": "lemma-workspace",
            "title": "External proof attempt",
            "authors": "Example Author",
            "year": "2026",
            "url": "https://example.com/collatz-proof",
            "source_type": "self_published",
            "claim_type": "proof_attempt",
            "summary": "Claims full proof by descent and cycle exclusion.",
            "fallacy_tags": ["empirical-not-proof"],
            "rubric": {
                "peer_reviewed": False,
                "acknowledged_errors": True,
                "defines_map_variant": False,
                "distinguishes_empirical_from_proof": False,
                "proves_descent": True,
                "proves_cycle_exclusion": True,
            },
        },
    )
    assert source_response.status_code == 200
    source_payload = source_response.json()
    assert source_payload["review_status"] == "intake"
    assert source_payload["fallacy_tags"] == ["empirical-not-proof"]

    review_response = client.post(
        f"/api/sources/{source_payload['id']}/review",
        json={
            "review_status": "flagged",
            "fallacy_tags": ["almost-all-not-all", "unchecked-generalization"],
            "notes": "Needs explicit map variant and a non-statistical universal argument.",
        },
    )
    assert review_response.status_code == 200
    assert review_response.json()["review_status"] == "flagged"

    baseline = client.get("/api/consensus-baseline")
    assert baseline.status_code == 200
    assert baseline.json()["problem_status"] == "open"

    probe = client.post(
        "/api/review/probes/modular",
        json={"modulus": 8, "allowed_residues": [5], "search_limit": 63},
    )
    assert probe.status_code == 200
    assert probe.json()["first_counterexample"] == 3

    catalog = client.get("/api/review/fallacy-tags")
    assert catalog.status_code == 200
    assert any(item["tag"] == "empirical-not-proof" for item in catalog.json())


def test_api_rejects_unknown_fallacy_tag(settings: Settings):
    app = create_app(settings)
    client = TestClient(app)

    response = client.post(
        "/api/sources",
        json={
            "direction_slug": "lemma-workspace",
            "title": "Invalid tags",
            "summary": "Should fail",
            "fallacy_tags": ["made-up-tag"],
        },
    )
    assert response.status_code == 400
    assert "Unknown fallacy tag" in response.json()["detail"]


def test_api_reddit_feed(settings: Settings, monkeypatch):
    app = create_app(settings)
    client = TestClient(app)

    def fake_feed(*, subreddit: str, sort: str, limit: int):
        assert subreddit == "Collatz"
        assert sort == "new"
        assert limit == 5
        return {
            "subreddit": "Collatz",
            "sort": "new",
            "fetched_at": "2026-03-21T00:00:00+00:00",
            "review_candidate_count": 1,
            "posts": [
                {
                    "id": "abc123",
                    "title": "Possible proof attempt",
                    "author": "example",
                    "permalink": "https://www.reddit.com/r/Collatz/comments/abc123/",
                    "created_at": "2026-03-21T00:00:00+00:00",
                    "score": 7,
                    "num_comments": 3,
                    "flair_text": "",
                    "signal": "review",
                    "excerpt": "Claims a proof.",
                }
            ],
        }

    monkeypatch.setattr("collatz_lab.api.fetch_subreddit_feed", fake_feed)
    response = client.get("/api/external/reddit/collatz?limit=5")
    assert response.status_code == 200
    payload = response.json()
    assert payload["review_candidate_count"] == 1
    assert payload["posts"][0]["signal"] == "review"
