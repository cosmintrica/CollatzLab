#!/usr/bin/env python3
"""
Local throughput benchmark: **cpu-sieve** vs **gpu-sieve** (MPS / CUDA).

Uses the same code path as the worker (`compute_range_metrics`). From repo root:

  cd /path/to/CollatzLab
  PYTHONPATH=backend/src python3 scripts/benchmark_mac_throughput.py

With venv (recommended on Mac for MPS):

  ./.venv/bin/python scripts/benchmark_mac_throughput.py

Options:
  --quick          small ranges (CI / smoke)
  --presets all    compare all MPS presets (default, tuned, extreme, maxbatch)
  --linear-end N   inclusive end for [1, N] (default 1_000_000 if --quick else 10_000_000)
  --hard-start S   use [S, S+width-1] instead of [1, width] (stress large n)
  --json           machine-readable output

MPS presets (gpu-sieve on Apple / same process only):
  default   — no env override (code default 1M / 128)
  tuned     — 1M batch, sync 128 (profile quick winner)
  extreme   — 2M batch, sync 128 (mac-dev-stack EXTREME)
  maxbatch  — 4M batch, sync 128 (large chunk, same sync)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any


def _bootstrap_path() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(root, "backend", "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


def _odd_count(start: int, end: int) -> int:
    first_odd = start if start & 1 else start + 1
    if first_odd > end:
        return 0
    return ((end - first_odd) // 2) + 1


def _apply_preset(name: str) -> dict[str, str]:
    """Return env key-values to merge (empty = leave process env)."""
    presets: dict[str, dict[str, str]] = {
        "default": {},
        "tuned": {
            "COLLATZ_MPS_SIEVE_BATCH_SIZE": "1048576",
            "COLLATZ_MPS_SYNC_EVERY": "128",
        },
        "extreme": {
            "COLLATZ_MPS_SIEVE_BATCH_SIZE": "2097152",
            "COLLATZ_MPS_SYNC_EVERY": "128",
        },
        "maxbatch": {
            "COLLATZ_MPS_SIEVE_BATCH_SIZE": "4194304",
            "COLLATZ_MPS_SYNC_EVERY": "128",
        },
    }
    if name not in presets:
        raise SystemExit(f"unknown preset {name!r}; choose from {sorted(presets)}")
    return presets[name]


def _run_timed(
    *,
    kernel: str,
    start: int,
    end: int,
    cpu_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """If ``cpu_reference`` is set (cpu-sieve aggregates), set ``parity_ok`` on this run."""
    from collatz_lab.services import compute_range_metrics

    t0 = time.perf_counter()
    agg = compute_range_metrics(start, end, kernel=kernel)
    elapsed = time.perf_counter() - t0
    odds = int(agg.processed)
    lin = end - start + 1
    out: dict[str, Any] = {
        "kernel": kernel,
        "start": start,
        "end": end,
        "linear_width": lin,
        "odd_seeds": odds,
        "seconds": round(elapsed, 6),
        "odd_per_sec": round(odds / elapsed, 0) if elapsed > 0 else 0.0,
        "linear_per_sec": round(lin / elapsed, 0) if elapsed > 0 else 0.0,
        "max_total": agg.max_total_stopping_time,
    }
    if cpu_reference is None:
        out["_parity"] = {
            "max_total_stopping_time": agg.max_total_stopping_time,
            "max_stopping_time": agg.max_stopping_time,
            "max_excursion": agg.max_excursion,
            "processed": agg.processed,
        }
    else:
        out["parity_ok"] = (
            agg.max_total_stopping_time == cpu_reference["max_total_stopping_time"]
            and agg.max_stopping_time == cpu_reference["max_stopping_time"]
            and agg.max_excursion == cpu_reference["max_excursion"]
            and agg.processed == cpu_reference["processed"]
        )
    return out


def main() -> int:
    _bootstrap_path()

    p = argparse.ArgumentParser(description="Benchmark cpu-sieve vs gpu-sieve throughput.")
    p.add_argument("--quick", action="store_true", help="Small ranges (~250k odds).")
    p.add_argument(
        "--linear-end",
        type=int,
        default=0,
        help="Inclusive end for [1, N] (default 1_000_000 if --quick else 10_000_000).",
    )
    p.add_argument(
        "--hard-start",
        type=int,
        default=0,
        help="If set, benchmark [S, S+width-1] instead of [1, width].",
    )
    p.add_argument(
        "--presets",
        default="default,extreme",
        help="Comma-separated MPS presets: default,tuned,extreme,maxbatch,all",
    )
    p.add_argument("--json", action="store_true", help="Print one JSON object to stdout.")
    p.add_argument("--no-warmup", action="store_true", help="Skip JIT/MPS warmup passes.")
    args = p.parse_args()

    if args.quick:
        width = 1_000_000
    else:
        width = args.linear_end if args.linear_end > 0 else 10_000_000

    start = 1
    end = width
    if args.hard_start > 0:
        start = args.hard_start
        end = start + width - 1

    preset_arg = args.presets.strip().lower()
    if preset_arg == "all":
        preset_names = ["default", "tuned", "extreme", "maxbatch"]
    else:
        preset_names = [x.strip() for x in preset_arg.split(",") if x.strip()]

    from collatz_lab.hardware import GPU_SIEVE_KERNEL, gpu_execution_ready
    from collatz_lab.hardware.gpu import cuda_gpu_execution_ready

    try:
        from collatz_lab import mps_collatz
    except Exception:
        mps_collatz = None

    mps_ok = bool(mps_collatz and mps_collatz.mps_accelerated_available())
    cuda_ok = cuda_gpu_execution_ready()
    gpu_ok = gpu_execution_ready()

    # MPS presets do not affect CUDA Numba; avoid duplicate gpu-sieve timings on NVIDIA.
    if cuda_ok and not mps_ok and preset_names != ["default"]:
        if not args.json:
            print("(CUDA without MPS: single gpu-sieve run; MPS presets ignored.)\n")
        preset_names = ["default"]

    saved_env = {k: os.environ.get(k) for k in ("COLLATZ_MPS_SIEVE_BATCH_SIZE", "COLLATZ_MPS_SYNC_EVERY")}

    def restore_env() -> None:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    rows: list[dict[str, Any]] = []

    def emit(row: dict[str, Any]) -> None:
        rows.append(row)

    if not args.json:
        print("== Collatz Lab — benchmark throughput ==")
        print(f"  Interval: [{start}, {end}]  (~{_odd_count(start, end):,} odd seeds)")
        print(f"  GPU ready={gpu_ok}  CUDA={cuda_ok}  MPS={mps_ok}")
        print()

    if not args.no_warmup:
        from collatz_lab.services import compute_range_metrics

        compute_range_metrics(1, 3000, kernel="cpu-sieve")
        if gpu_ok:
            compute_range_metrics(1, 3000, kernel=GPU_SIEVE_KERNEL)

    cpu_row = _run_timed(kernel="cpu-sieve", start=start, end=end)
    cpu_row["preset"] = None
    cpu_parity = cpu_row.pop("_parity")
    emit(cpu_row)
    if not args.json:
        print(
            f"cpu-sieve     {cpu_row['seconds']:>10.4f}s  "
            f"{cpu_row['odd_per_sec']/1e6:>6.2f} M odd/s  "
            f"({cpu_row['linear_per_sec']/1e6:.2f} M linear/s)"
        )

    if gpu_ok:
        for pname in preset_names:
            updates = _apply_preset(pname)
            restore_env()
            for k, v in updates.items():
                os.environ[k] = v
            g_row = _run_timed(
                kernel=GPU_SIEVE_KERNEL,
                start=start,
                end=end,
                cpu_reference=cpu_parity,
            )
            g_row["preset"] = pname
            emit(g_row)
            if not args.json:
                tag = f"gpu-sieve[{pname}]"
                parity = "OK" if g_row.get("parity_ok") else "MISMATCH"
                print(
                    f"{tag:<22} {g_row['seconds']:>10.4f}s  "
                    f"{g_row['odd_per_sec']/1e6:>6.2f} M odd/s  "
                    f"({g_row['linear_per_sec']/1e6:.2f} M linear/s)  parity={parity}"
                )
        restore_env()
    else:
        if not args.json:
            print("\n(gpu-sieve skipped — no GPU backend)")

    if not args.json:
        print()
        best = max((r for r in rows if r.get("kernel") == GPU_SIEVE_KERNEL), key=lambda x: x["odd_per_sec"], default=None)
        if best:
            ratio = best["odd_per_sec"] / cpu_row["odd_per_sec"] if cpu_row["odd_per_sec"] else 0
            print(f"Best gpu preset on this machine: {best.get('preset')}  (~{ratio:.2f}× vs cpu-sieve for this range).")
        print("Tip: export winning COLLATZ_MPS_* for the worker, or COLLATZ_MAC_EXTREME_THROUGHPUT=1 with mac-dev-stack.")
        print("     For a full grid sweep run: scripts/profile_mps_metal_sieve.py")

    if args.json:
        print(json.dumps({"interval": {"start": start, "end": end}, "runs": rows}, indent=2))

    mismatches = [r for r in rows if r.get("parity_ok") is False]
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
