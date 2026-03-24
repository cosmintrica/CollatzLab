from __future__ import annotations

import json
import platform

import pytest
from fastapi.testclient import TestClient

from collatz_lab.api import create_app
from collatz_lab.config import Settings


def test_bench_metal_chunk_status(settings: Settings) -> None:
    app = create_app(settings)
    client = TestClient(app)
    r = client.get("/api/bench/metal-chunk/status")
    assert r.status_code == 200
    body = r.json()
    assert body["darwin"] == (platform.system() == "Darwin")
    assert body.get("server_system") == platform.system()
    assert "metal_available" in body
    assert "active_job" in body


@pytest.mark.skipif(platform.system() == "Darwin", reason="Avoid queueing a real Metal job on macOS CI/dev.")
def test_bench_metal_chunk_run_rejects_non_macos(settings: Settings) -> None:
    app = create_app(settings)
    client = TestClient(app)
    r = client.post("/api/bench/metal-chunk/run", json={"quick": True})
    assert r.status_code == 400
    assert "macOS" in r.json()["detail"]


def test_repository_metal_benchmark_roundtrip(repository) -> None:
    repository.save_metal_benchmark_run(
        job_id="bench-test-1",
        created_at="2020-01-01T00:00:00+00:00",
        finished_at="2020-01-01T00:01:00+00:00",
        status="completed",
        params_json=json.dumps({"quick": True}),
        result_json=json.dumps(
            {
                "winner": {"metal_chunk_size": 1_048_576, "odd_per_sec_millions": 12.3},
                "small_range_parity_ok": True,
                "calibration_written": "/tmp/x",
            }
        ),
        error_message="",
    )
    rows = repository.list_metal_benchmark_runs(limit=5)
    assert len(rows) == 1
    assert rows[0]["id"] == "bench-test-1"
    assert rows[0]["summary"]["winner_chunk"] == 1_048_576
    detail = repository.get_metal_benchmark_run("bench-test-1")
    assert detail is not None
    assert detail["params"]["quick"] is True
    assert detail["result"]["winner"]["odd_per_sec_millions"] == 12.3


def test_repository_metal_benchmark_hall_of_fame_sorts_by_throughput(repository) -> None:
    def save(jid: str, mps: float) -> None:
        repository.save_metal_benchmark_run(
            job_id=jid,
            created_at="2020-01-01T00:00:00+00:00",
            finished_at="2020-01-02T00:00:00+00:00",
            status="completed",
            params_json=json.dumps({"quick": True}),
            result_json=json.dumps(
                {
                    "platform": "Darwin",
                    "winner": {"metal_chunk_size": 2_097_152, "odd_per_sec_millions": mps},
                    "interval": {"start": 1, "end": 12_000_000, "odd_seeds": 6_000_000},
                    "small_range_parity_ok": True,
                }
            ),
            error_message="",
        )

    save("slower", 8.5)
    save("faster", 22.1)
    hof = repository.list_metal_benchmark_hall_of_fame(platform="Darwin", limit=10)
    assert [row["id"] for row in hof] == ["faster", "slower"]
    assert hof[0]["rank"] == 1
    assert hof[0]["throughput_m_per_s"] == 22.1


def test_bench_hall_of_fame_api_bad_platform(settings: Settings) -> None:
    client = TestClient(create_app(settings))
    r = client.get("/api/bench/metal-chunk/hall-of-fame?platform=InvalidOS")
    assert r.status_code == 400


def test_bench_metal_chunk_presets_list(settings: Settings) -> None:
    client = TestClient(create_app(settings))
    r = client.get("/api/bench/metal-chunk/presets")
    assert r.status_code == 200
    ids = {item["id"] for item in r.json()}
    assert ids >= {"standard", "extended", "custom"}


def test_resolve_metal_bench_preset_standard_overrides_client_fields() -> None:
    from collatz_lab.bench_metal_presets import resolve_metal_bench_params

    out = resolve_metal_bench_params(
        preset="standard",
        quick=False,
        linear_end=99,
        reps=9,
        warmup=0,
        chunks_csv="1,2,3",
        write_calibration=False,
        pipeline_ab=True,
    )
    assert out["preset"] == "standard"
    assert out["quick"] is True
    assert out["linear_end"] == 24_000_000
    assert out["reps"] == 5
    assert out["warmup"] == 2
    assert out["chunks_csv"] == ""
    assert out["write_calibration"] is False
    assert out["pipeline_ab"] is True
