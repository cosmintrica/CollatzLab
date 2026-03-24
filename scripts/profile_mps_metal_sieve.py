#!/usr/bin/env python3
"""
Metal / PyTorch MPS sweep for **gpu-sieve** before changing production defaults.

Measures wall time for ``compute_range_metrics`` across ``COLLATZ_MPS_SIEVE_BATCH_SIZE``
and ``COLLATZ_MPS_SYNC_EVERY``. Run on Apple Silicon with the project venv:

  cd CollatzLab
  PYTHONPATH=backend/src ./.venv/bin/python scripts/profile_mps_metal_sieve.py

Use ``--json`` for machine-readable output. Increase ``--linear-width`` once combos look stable.

**Why it “hangs”:** the full grid is **16** gpu-sieve passes on ``[1, N]`` plus warmups. For ``N`` in the millions, each pass can take **minutes** on MPS, so the full sweep can exceed **30–60+ minutes**. Use ``--quick`` first (4 combos, smaller ``N``). IDE terminals may also **abort** long jobs before they finish.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any


def _bootstrap_path() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(root, "backend", "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


def main() -> int:
    _bootstrap_path()

    p = argparse.ArgumentParser(description="Sweep MPS env knobs for gpu-sieve.")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Small grid (4 combos) and N=800k — finishes in a few minutes on typical M1/M2/M3.",
    )
    p.add_argument("--linear-width", type=int, default=0, help="Interval [1, N] (default: 4M full, 800k with --quick).")
    p.add_argument("--reps", type=int, default=2, help="Timed repetitions per combo (after warmup).")
    p.add_argument("--warmup", type=int, default=1, help="Untimed warmup runs per combo.")
    p.add_argument("--json", action="store_true", help="Print JSON only.")
    args = p.parse_args()

    try:
        from collatz_lab import mps_collatz
        from collatz_lab.hardware import GPU_SIEVE_KERNEL
        from collatz_lab.services import compute_range_metrics
    except Exception as exc:  # pragma: no cover
        print(f"import failed: {exc}", file=sys.stderr)
        return 2

    if not (mps_collatz and mps_collatz.mps_accelerated_available()):
        print("MPS not available (need Apple Silicon + torch MPS). Skipping.", file=sys.stderr)
        return 0

    if args.quick:
        end = max(1000, args.linear_width or 800_000)
        batch_grid = [1_048_576, 2_097_152]
        sync_grid = [128, 256]
    else:
        end = max(1000, args.linear_width or 4_000_000)
        batch_grid = [524_288, 1_048_576, 2_097_152, 4_194_304]
        sync_grid = [64, 128, 256, 512]

    # Parity vs cpu-sieve once (small range) under default env
    for k in ("COLLATZ_MPS_SIEVE_BATCH_SIZE", "COLLATZ_MPS_SYNC_EVERY"):
        os.environ.pop(k, None)
    ref = compute_range_metrics(1, 8000, kernel="cpu-sieve")
    gpu_chk = compute_range_metrics(1, 8000, kernel=GPU_SIEVE_KERNEL)
    parity_ok = (
        gpu_chk.max_total_stopping_time == ref.max_total_stopping_time
        and gpu_chk.processed == ref.processed
    )
    n_combos = len(batch_grid) * len(sync_grid)
    if not args.json:
        print("== MPS gpu-sieve profile ==", flush=True)
        print(
            f"  interval [1, {end}]  combos={n_combos}  parity_small_range={'OK' if parity_ok else 'FAIL'}",
            flush=True,
        )
        if n_combos > 8:
            print(
                "  note: full grid can take 30–90+ minutes on MPS; use --quick for a short run.",
                flush=True,
            )
        if not parity_ok:
            print("  aborting sweep until parity is fixed", file=sys.stderr, flush=True)
            return 1
        print(flush=True)

    saved = {k: os.environ.get(k) for k in ("COLLATZ_MPS_SIEVE_BATCH_SIZE", "COLLATZ_MPS_SYNC_EVERY")}
    rows: list[dict[str, Any]] = []

    combo_i = 0
    try:
        for batch in batch_grid:
            for sync in sync_grid:
                combo_i += 1
                if not args.json:
                    print(
                        f"  [{combo_i}/{n_combos}] batch={batch} sync={sync} — warmup…",
                        flush=True,
                    )
                os.environ["COLLATZ_MPS_SIEVE_BATCH_SIZE"] = str(batch)
                os.environ["COLLATZ_MPS_SYNC_EVERY"] = str(sync)
                for _ in range(max(0, args.warmup)):
                    compute_range_metrics(1, min(50_000, end), kernel=GPU_SIEVE_KERNEL)
                if not args.json:
                    print(f"  [{combo_i}/{n_combos}] timing {args.reps} rep(s) on [1, {end}]…", flush=True)
                times: list[float] = []
                for _ in range(max(1, args.reps)):
                    t0 = time.perf_counter()
                    compute_range_metrics(1, end, kernel=GPU_SIEVE_KERNEL)
                    times.append(time.perf_counter() - t0)
                med = float(statistics.median(times))
                odds = (end // 2) if end > 1 else 1
                rows.append(
                    {
                        "mps_sieve_batch_size": batch,
                        "mps_sync_every": sync,
                        "median_seconds": round(med, 4),
                        "odd_per_sec": round(odds / med, 0) if med > 0 else 0.0,
                    }
                )
                if not args.json:
                    print(
                        f"  batch={batch:7d} sync={sync:3d}  "
                        f"median={med:8.3f}s  ~{rows[-1]['odd_per_sec']/1e6:.3f} M odd/s",
                        flush=True,
                    )
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    best = min(rows, key=lambda r: r["median_seconds"]) if rows else None
    if not args.json:
        print()
        if best:
            print(
                "Best combo (lowest median wall time): "
                f"COLLATZ_MPS_SIEVE_BATCH_SIZE={best['mps_sieve_batch_size']} "
                f"COLLATZ_MPS_SYNC_EVERY={best['mps_sync_every']}"
            )
    else:
        print(json.dumps({"interval_end": end, "parity_small_range": parity_ok, "rows": rows, "best": best}, indent=2))

    return 0 if parity_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
