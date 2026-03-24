#!/usr/bin/env python3
"""
Estimate **stdio JSON round-trip + small GPU chunk** cost vs a **large** chunk.

This does **not** implement a `.dylib` — it bounds how much wall time might be saved if the
Python↔helper boundary disappeared entirely (upper bound ≈ round_trips × median_small_t).

From repo root (macOS + built ``metal_sieve_chunk``):

  export KMP_DUPLICATE_LIB_OK=TRUE
  PYTHONPATH=backend/src ./.venv/bin/python scripts/benchmark_metal_stdio_overhead.py

Interpretation: if a full run uses ``R`` chunks of size ``C`` and GPU time dominates, moving to
in-process calls removes roughly ``R × t_small`` of framing/sync overhead **only** — not a
substitute for profiling the real ``profile_metal_sieve_chunk.py`` sweep.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time


def _bootstrap_path() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(root, "backend", "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


def main() -> int:
    _bootstrap_path()
    p = argparse.ArgumentParser(description="Metal stdio overhead micro-benchmark.")
    p.add_argument("--trips", type=int, default=40, help="Timed round-trips per size.")
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    if sys.platform != "darwin":
        print("macOS + Metal helper only.", file=sys.stderr)
        return 2

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ["COLLATZ_GPU_SIEVE_BACKEND"] = "metal"

    from collatz_lab.gpu_sieve_metal_runtime import (
        metal_sieve_chunk_binary_path,
        native_metal_sieve_available,
        shutdown_metal_stdio_transport,
    )
    from collatz_lab.gpu_sieve_metal_runtime import _run_metal_chunk  # noqa: PLC2701

    if not native_metal_sieve_available():
        print("metal_sieve_chunk not available.", file=sys.stderr)
        return 2

    bp = metal_sieve_chunk_binary_path()
    if bp is None:
        return 2

    shutdown_metal_stdio_transport()

    sizes = (2048, 65_536, 1_048_576)
    rows: list[dict] = []

    for count in sizes:
        for _ in range(args.warmup):
            _run_metal_chunk(bp, 1, count)
        samples: list[float] = []
        for _ in range(args.trips):
            t0 = time.perf_counter()
            _run_metal_chunk(bp, 1, count)
            samples.append(time.perf_counter() - t0)
        med = float(statistics.median(samples))
        per_odd_us = (med / count) * 1e6 if count else 0.0
        rows.append(
            {
                "odd_count": count,
                "seconds_median": round(med, 6),
                "us_per_odd_median": round(per_odd_us, 4),
            }
        )

    small = rows[0]["seconds_median"]
    large = rows[-1]["seconds_median"]
    large_n = rows[-1]["odd_count"]
    # Naive upper bound: if all work were like the tiny chunk, overhead dominates; compare slopes.
    overhead_guess = max(0.0, small - (large / large_n) * rows[0]["odd_count"])

    out = {
        "helper": str(bp),
        "trips": args.trips,
        "per_size": rows,
        "note": (
            "small_chunk_median_s is mostly JSON+sync+tiny GPU; "
            "large row shows amortized cost per chunk. "
            "dylib would shave some of the fixed per-call cost, not Collatz math itself."
        ),
        "rough_fixed_cost_seconds_guess": round(overhead_guess, 6),
    }

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print("== Metal stdio micro-benchmark ==")
        print(f"  helper: {bp}")
        for r in rows:
            print(
                f"  count={r['odd_count']:>8}  median={r['seconds_median']:.5f}s  "
                f"~{r['us_per_odd_median']:.3f} µs/odd"
            )
        print()
        print(f"  Rough fixed-ish cost vs scaling (heuristic): ~{overhead_guess*1000:.2f} ms per small-chunk trip")
        print("  See docs/METAL_DYLIB_ROADMAP.md — full dylib still needed to remove subprocess/pipes.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
