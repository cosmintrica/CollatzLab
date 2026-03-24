"""Default compute profile seeding (GPU lane 0% on macOS for new DBs)."""

from __future__ import annotations

import platform


def test_default_compute_profile_seed_payload_darwin(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    from collatz_lab.repository import default_compute_profile_seed_payload

    p = default_compute_profile_seed_payload()
    assert p["gpu_percent"] == 0
    assert p["cpu_percent"] == 100
    assert p["system_percent"] == 100


def test_default_compute_profile_seed_payload_linux(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    from collatz_lab.repository import default_compute_profile_seed_payload

    p = default_compute_profile_seed_payload()
    assert p["gpu_percent"] == 100
