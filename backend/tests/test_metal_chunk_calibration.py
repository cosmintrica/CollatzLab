"""Calibration file for Metal chunk auto-tuning."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


def _write_cal(path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "chunk_odds": 8_388_608,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "platform": "darwin",
        "interval": {"start": 1, "end": 12_000_000},
        "winner": {"metal_chunk_size": 8_388_608},
        "odd_per_sec_millions": 420.0,
        "seconds_median": 0.014,
        **kwargs,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_choose_prefers_calibration_when_fresh(monkeypatch, tmp_path: Path):
    from collatz_lab import metal_chunk_calibration as cal

    monkeypatch.setenv("COLLATZ_LAB_ROOT", str(tmp_path))
    monkeypatch.setenv("COLLATZ_METAL_SIEVE_USE_CALIBRATION", "1")
    _write_cal(cal.metal_chunk_calibration_path())
    picked = cal.choose_chunk_from_calibration(cap=16_777_216, max_odds_ram=16_777_216)
    assert picked == 8_388_608


def test_choose_clamps_to_ram_ceiling(monkeypatch, tmp_path: Path):
    from collatz_lab import metal_chunk_calibration as cal

    monkeypatch.setenv("COLLATZ_LAB_ROOT", str(tmp_path))
    monkeypatch.setenv("COLLATZ_METAL_SIEVE_USE_CALIBRATION", "1")
    _write_cal(cal.metal_chunk_calibration_path(), chunk_odds=16_777_216)
    picked = cal.choose_chunk_from_calibration(cap=16_777_216, max_odds_ram=4_194_304)
    assert picked == 4_194_304


def test_stale_calibration_ignored(monkeypatch, tmp_path: Path):
    from collatz_lab import metal_chunk_calibration as cal

    monkeypatch.setenv("COLLATZ_LAB_ROOT", str(tmp_path))
    monkeypatch.setenv("COLLATZ_METAL_SIEVE_USE_CALIBRATION", "1")
    monkeypatch.setenv("COLLATZ_METAL_SIEVE_CALIBRATION_MAX_AGE_DAYS", "1")
    old = (datetime.now(timezone.utc) - timedelta(days=9)).isoformat()
    _write_cal(cal.metal_chunk_calibration_path(), recorded_at=old)
    assert cal.choose_chunk_from_calibration(cap=16_777_216, max_odds_ram=16_777_216) is None
