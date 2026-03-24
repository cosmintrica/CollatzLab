"""
Canonical, reproducible Metal chunk-benchmark **presets** (same inputs on every machine).

Wall-clock duration still varies by hardware; aggregate JSON remains comparable when the same
preset and ``metal_sieve_chunk`` build are used. Chunk sizes above the runtime RAM cap are
filtered out server-side (see ``run_metal_chunk_benchmark``).
"""

from __future__ import annotations

from typing import Any, Final, Literal

PresetId = Literal["standard", "extended", "custom"]

# Fixed odd-count chunk ladder (odds per Metal dispatch). Intentionally identical repo-wide.
CANONICAL_CHUNK_ODDS: Final[tuple[int, ...]] = (
    1_048_576,
    2_097_152,
    4_194_304,
    8_388_608,
    16_777_216,
)

# Human-readable spec for dashboards / papers (interval is linear inclusive end when linear_end=0).
# Longer defaults than the original 12M/48M quick paths so wall-time sweeps stabilize
# thermals and timing noise; still one canonical ladder for cross-machine comparison.
_PRESET_PUBLIC: dict[str, dict[str, Any]] = {
    "standard": {
        "id": "standard",
        "title": "Standard (reproducible)",
        "summary": "Interval [1, 24M] linear; canonical 1M→16M odds ladder; 5 timed reps, 2 warmup.",
        "interval_linear_end": 24_000_000,
        "chunk_odds": list(CANONICAL_CHUNK_ODDS),
        "reps": 5,
        "warmup": 2,
    },
    "extended": {
        "id": "extended",
        "title": "Extended (reproducible)",
        "summary": "Interval [1, 96M] linear; same chunk ladder; 5 timed reps, 2 warmup.",
        "interval_linear_end": 96_000_000,
        "chunk_odds": list(CANONICAL_CHUNK_ODDS),
        "reps": 5,
        "warmup": 2,
    },
    "custom": {
        "id": "custom",
        "title": "Custom (advanced)",
        "summary": "You control interval, reps, warmup, and optional chunk CSV. Not comparable across labs unless you publish the exact body.",
        "interval_linear_end": None,
        "chunk_odds": None,
        "reps": None,
        "warmup": None,
    },
}


def list_metal_bench_presets_public() -> list[dict[str, Any]]:
    return [
        _PRESET_PUBLIC["standard"],
        _PRESET_PUBLIC["extended"],
        _PRESET_PUBLIC["custom"],
    ]


def resolve_metal_bench_params(
    *,
    preset: str,
    quick: bool,
    linear_end: int,
    reps: int,
    warmup: int,
    chunks_csv: str,
    write_calibration: bool,
    pipeline_ab: bool,
) -> dict[str, Any]:
    """Merge UI/API fields with a preset. ``custom`` keeps caller-supplied tuning fields."""
    pid = (preset or "standard").strip().lower()
    if pid not in ("standard", "extended", "custom"):
        pid = "standard"

    if pid == "custom":
        return {
            "preset": "custom",
            "quick": quick,
            "linear_end": linear_end,
            "reps": max(1, min(12, reps)),
            "warmup": max(0, min(5, warmup)),
            "chunks_csv": chunks_csv.strip(),
            "write_calibration": write_calibration,
            "pipeline_ab": pipeline_ab,
        }

    if pid == "standard":
        return {
            "preset": "standard",
            "quick": True,
            "linear_end": 24_000_000,
            "reps": 5,
            "warmup": 2,
            "chunks_csv": "",
            "write_calibration": write_calibration,
            "pipeline_ab": pipeline_ab,
        }

    # extended
    return {
        "preset": "extended",
        "quick": False,
        "linear_end": 96_000_000,
        "reps": 5,
        "warmup": 2,
        "chunks_csv": "",
        "write_calibration": write_calibration,
        "pipeline_ab": pipeline_ab,
    }
