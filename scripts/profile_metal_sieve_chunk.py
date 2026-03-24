#!/usr/bin/env python3
"""
Sweep **COLLATZ_METAL_SIEVE_CHUNK_SIZE** for native Metal ``gpu-sieve`` (``metal_sieve_chunk``).

Measures wall time and M odd/s on a fixed interval while varying only the Metal chunk size.
Use from repo root with the project venv (Metal + built binary required):

  cd CollatzLab
  PYTHONPATH=backend/src ./.venv/bin/python scripts/profile_metal_sieve_chunk.py --quick
  PYTHONPATH=backend/src ./.venv/bin/python scripts/profile_metal_sieve_chunk.py --json

Env:
  COLLATZ_GPU_SIEVE_BACKEND=metal  (set automatically for this script)
  COLLATZ_METAL_SIEVE_STDIO_PIPELINE=0|1  (optional A/B for Swift stdio overlap)

**RAM / swap:** larger chunks grow the helper's grow-only ``steps`` buffer (~4 bytes × largest chunk).
On 16 GB machines with heavy swap, prefer ``--quick`` or smaller ``--linear-end``.

If the process aborts with **OpenMP duplicate libomp** (native CPU ``.dylib`` + Numba), prefix:

``KMP_DUPLICATE_LIB_OK=TRUE`` (unsupported workaround; see LLVM OpenMP docs).

**Throughput-first auto:** ``--write-calibration`` saves the sweep winner to
``data/metal_sieve_chunk_calibration.json`` so workers pick it when
``COLLATZ_METAL_SIEVE_CHUNK_SIZE`` is unset (still RAM-clamped).

Implementation: ``collatz_lab.bench_metal_chunk.run_metal_chunk_benchmark`` (shared with API/UI).
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _bootstrap_path() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(root, "backend", "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


def main() -> int:
    _bootstrap_path()

    p = argparse.ArgumentParser(description="Profile Metal gpu-sieve chunk sizes.")
    p.add_argument("--quick", action="store_true", help="Smaller interval (~6M odds).")
    p.add_argument(
        "--linear-end",
        type=int,
        default=0,
        help="Inclusive end for [1, N] (default 12M or 48M linear).",
    )
    p.add_argument("--reps", type=int, default=2, help="Timed repetitions per chunk size (after warmup).")
    p.add_argument("--warmup", type=int, default=1, help="Untimed warmup runs per chunk size.")
    p.add_argument(
        "--chunks",
        default="",
        help="Comma-separated chunk sizes (odds). Default: sweep grid within runtime max.",
    )
    p.add_argument("--json", action="store_true", help="Print one JSON object only.")
    p.add_argument(
        "--print-auto",
        action="store_true",
        help="Print resolved auto chunk (COLLATZ_METAL_SIEVE_CHUNK_AUTO + psutil); exit without Metal run.",
    )
    p.add_argument(
        "--pipeline-ab",
        action="store_true",
        help="Also compare COLLATZ_METAL_SIEVE_STDIO_PIPELINE=0 vs 1 at best chunk size.",
    )
    p.add_argument(
        "--write-calibration",
        action="store_true",
        help="Write data/metal_sieve_chunk_calibration.json from sweep winner (throughput-first auto).",
    )
    args = p.parse_args()

    if sys.platform != "darwin":
        if args.json:
            print(json.dumps({"error": "darwin_only", "platform": sys.platform}))
        else:
            print("Metal chunk profiling requires macOS.", file=sys.stderr)
        return 2

    os.environ["COLLATZ_GPU_SIEVE_BACKEND"] = "metal"
    for k in ("COLLATZ_MPS_SIEVE_BATCH_SIZE", "COLLATZ_MPS_SYNC_EVERY"):
        os.environ.pop(k, None)

    if args.print_auto:
        from collatz_lab.gpu_sieve_metal_runtime import (
            metal_sieve_chunk_auto_enabled,
            metal_sieve_chunk_max_odds,
            resolve_metal_sieve_chunk_odds,
        )
        from collatz_lab.metal_chunk_calibration import get_metal_chunk_calibration_status

        auto = {
            "metal_chunk_auto_enabled": metal_sieve_chunk_auto_enabled(),
            "metal_chunk_max_odds": metal_sieve_chunk_max_odds(),
            "metal_chunk_resolved_odds": resolve_metal_sieve_chunk_odds(),
            "env_COLLATZ_METAL_SIEVE_CHUNK_SIZE": os.getenv("COLLATZ_METAL_SIEVE_CHUNK_SIZE"),
            "env_COLLATZ_METAL_SIEVE_CHUNK_AUTO": os.getenv("COLLATZ_METAL_SIEVE_CHUNK_AUTO", "1"),
        }
        auto.update(get_metal_chunk_calibration_status())
        if args.json:
            print(json.dumps(auto, indent=2))
        else:
            print("== Metal chunk auto (no GPU sweep) ==")
            for k, v in auto.items():
                print(f"  {k}: {v}")
        return 0

    from collatz_lab.bench_metal_chunk import run_metal_chunk_benchmark
    from collatz_lab.gpu_sieve_metal_runtime import metal_sieve_chunk_binary_path

    try:
        out = run_metal_chunk_benchmark(
            quick=args.quick,
            linear_end=args.linear_end,
            reps=args.reps,
            warmup=args.warmup,
            chunks_csv=args.chunks,
            write_calibration=args.write_calibration,
            pipeline_ab=args.pipeline_ab,
        )
    except RuntimeError as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}))
        else:
            print(str(exc), file=sys.stderr)
        return 2

    if not out["small_range_parity_ok"] and not args.json:
        print(
            "warning: small-range parity vs cpu-sieve failed — still reporting sweep.",
            file=sys.stderr,
        )

    if args.json:
        # CLI historically omitted helper_path in JSON; keep parity with shared dict (includes helper_path).
        print(json.dumps(out, indent=2))
        return 0

    bp = out.get("helper_path") or metal_sieve_chunk_binary_path()
    interval = out["interval"]
    start, end = interval["start"], interval["end"]
    odds = interval["odd_seeds"]
    chunk_grid = [r["metal_chunk_size"] for r in out["runs"]]
    print("== Metal gpu-sieve — chunk size sweep ==")
    print(f"  helper: {bp}")
    print(f"  interval: [{start}, {end}]  (~{odds:,} odd seeds)")
    print(f"  runtime max chunk (env test): {out['runtime_max_chunk']:,}")
    print(f"  chunk grid: {[f'{c:,}' for c in chunk_grid]}")
    print(f"  reps={args.reps} warmup={args.warmup}")
    print()
    for r in out["runs"]:
        print(
            f"  chunk={r['metal_chunk_size']:>12,}  median={r['seconds_median']:>10.4f}s  "
            f"{r['odd_per_sec_millions']:>8.2f} M odd/s"
        )
    best = out["winner"]
    if out.get("pipeline_ab"):
        print("\n== STDIO pipeline A/B (best chunk) ==")
        for pr in out["pipeline_ab"]:
            pl = "1" if pr["stdio_pipeline"] else "0"
            print(
                f"  pipeline={pl}  median={pr['seconds_median']:.4f}s  "
                f"{pr['odd_per_sec_millions']:.2f} M odd/s"
            )
    print()
    print(
        f"Best: chunk={best['metal_chunk_size']:,}  "
        f"{best['odd_per_sec_millions']:.2f} M odd/s  ({best['seconds_median']:.4f}s median)"
    )
    print("Export for worker:  export COLLATZ_METAL_SIEVE_CHUNK_SIZE=<winner>")
    print("Optional:           export COLLATZ_METAL_SIEVE_STDIO_PIPELINE=1")
    cal = out.get("calibration_written")
    if cal:
        print(f"Calibration file:   {cal}")
        print("  Restart workers so auto picks this chunk (still RAM-clamped).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
