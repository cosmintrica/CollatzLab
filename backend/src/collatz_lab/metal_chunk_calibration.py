"""
Persisted **Metal chunk** calibration from a local throughput sweep.

When ``COLLATZ_METAL_SIEVE_CHUNK_SIZE`` is unset and auto-tuning is on (macOS), the worker
prefers the calibrated ``chunk_odds`` (clamped by RAM safety), then falls back to the
memory ladder. Users refresh calibration after OS / PyTorch / Metal helper upgrades.

File: ``${COLLATZ_LAB_ROOT}/data/metal_sieve_chunk_calibration.json`` (created by
``scripts/profile_metal_sieve_chunk.py --write-calibration``).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CALIBRATION_VERSION = 1
DEFAULT_FILENAME = "metal_sieve_chunk_calibration.json"


def metal_chunk_calibration_path() -> Path:
    root = Path(os.getenv("COLLATZ_LAB_ROOT", os.getcwd())).resolve()
    return root / "data" / DEFAULT_FILENAME


def calibration_max_age_days() -> float:
    try:
        return float(os.getenv("COLLATZ_METAL_SIEVE_CALIBRATION_MAX_AGE_DAYS", "30"))
    except ValueError:
        return 30.0


def calibration_enabled() -> bool:
    raw = os.getenv("COLLATZ_METAL_SIEVE_USE_CALIBRATION", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _parse_iso(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def read_calibration_raw() -> dict[str, Any] | None:
    path = metal_chunk_calibration_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or int(data.get("version", 0)) != CALIBRATION_VERSION:
        return None
    if data.get("platform") and data["platform"] != "darwin":
        return None
    co = data.get("chunk_odds")
    if co is None:
        return None
    try:
        co_int = int(co)
    except (TypeError, ValueError):
        return None
    if co_int < 4096:
        return None
    out = dict(data)
    out["_parsed_chunk_odds"] = co_int
    return out


def calibration_age_days(raw: dict[str, Any]) -> float | None:
    ts = raw.get("recorded_at")
    if not ts or not isinstance(ts, str):
        return None
    then = _parse_iso(ts)
    if then is None:
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - then
    return max(0.0, delta.total_seconds() / 86400.0)


def calibration_is_fresh(raw: dict[str, Any]) -> bool:
    age = calibration_age_days(raw)
    if age is None:
        return False
    return age <= calibration_max_age_days()


def choose_chunk_from_calibration(cap: int, max_odds_ram: int) -> int | None:
    """Return calibrated chunk odds if file is valid and fresh; always clamped to ``cap`` and RAM."""
    if not calibration_enabled():
        return None
    raw = read_calibration_raw()
    if raw is None or not calibration_is_fresh(raw):
        return None
    co = int(raw["_parsed_chunk_odds"])
    chosen = max(4096, min(cap, co, max_odds_ram))
    return chosen


def get_metal_chunk_calibration_status() -> dict[str, Any]:
    """For API diagnostics (safe on all platforms)."""
    path = metal_chunk_calibration_path()
    raw = read_calibration_raw()
    if raw is None:
        return {
            "calibration_path": str(path),
            "calibration_present": False,
            "calibration_fresh": False,
            "calibration_enabled": calibration_enabled(),
        }
    age = calibration_age_days(raw)
    fresh = calibration_is_fresh(raw)
    return {
        "calibration_path": str(path),
        "calibration_present": True,
        "calibration_fresh": fresh,
        "calibration_enabled": calibration_enabled(),
        "calibration_age_days": round(age, 2) if age is not None else None,
        "calibration_chunk_odds": raw.get("chunk_odds"),
        "calibration_recorded_at": raw.get("recorded_at"),
        "calibration_odd_per_sec_millions": raw.get("odd_per_sec_millions"),
    }


def write_metal_chunk_calibration(
    *,
    chunk_odds: int,
    interval: dict[str, Any],
    winner_row: dict[str, Any],
    platform: str,
) -> Path:
    """Write calibration JSON; creates ``data/`` if needed."""
    path = metal_chunk_calibration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CALIBRATION_VERSION,
        "chunk_odds": int(chunk_odds),
        "recorded_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "platform": platform,
        "interval": interval,
        "winner": winner_row,
        "odd_per_sec_millions": winner_row.get("odd_per_sec_millions"),
        "seconds_median": winner_row.get("seconds_median"),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
