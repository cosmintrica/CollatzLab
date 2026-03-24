#!/usr/bin/env python3
"""Wall-clock Numba vs native (auto) for the same ``cpu-sieve`` interval.

Run from the repo root or anywhere with the backend dependencies on PYTHONPATH.

Example:
  python3 scripts/native_sieve_kit/benchmark_cpu_sieve_numba_vs_native.py --start 1 --end 500000

Prints JSON: ``seconds_numba``, ``seconds_resolved`` (auto = native when .dylib/.so is found),
``resolved_backend``, ``openmp_linked``, ``speedup_numba_over_resolved``.

Before timing, runs a **warmup** on ``[1, min(100_000, --end)]`` when ``--end > 20_000`` so
``seconds_numba`` excludes Numba JIT compile time; native has no JIT but uses the same warmup for a fair comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark cpu-sieve Numba vs native (auto).")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=400_000)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    backend_src = root / "backend" / "src"
    sys.path.insert(0, str(backend_src))
    os.environ.setdefault("COLLATZ_LAB_ROOT", str(root))

    from collatz_lab.hardware import CPU_SIEVE_KERNEL
    from collatz_lab.cpu_sieve_native_runtime import (
        clear_cpu_sieve_native_runtime_caches,
        cpu_sieve_resolve_backend,
        native_cpu_sieve_openmp_linked,
    )
    from collatz_lab.services import compute_range_metrics

    def timed_run(mode: str) -> float:
        os.environ["COLLATZ_CPU_SIEVE_BACKEND"] = mode
        clear_cpu_sieve_native_runtime_caches()
        warmup_note: str | None = None
        if args.end > 20_000:
            warm_end = min(100_000, args.end)
            compute_range_metrics(1, warm_end, kernel=CPU_SIEVE_KERNEL)
            warmup_note = f"1..{warm_end}"
        t0 = time.perf_counter()
        compute_range_metrics(
            args.start,
            args.end,
            kernel=CPU_SIEVE_KERNEL,
        )
        dt = time.perf_counter() - t0
        return dt, warmup_note

    t_numba, warm_n = timed_run("numba")
    clear_cpu_sieve_native_runtime_caches()
    os.environ.pop("COLLATZ_CPU_SIEVE_BACKEND", None)
    # default auto
    resolved = cpu_sieve_resolve_backend()
    t_resolved, warm_a = timed_run("auto")
    omp = native_cpu_sieve_openmp_linked()

    out = {
        "start": args.start,
        "end": args.end,
        "warmup_range_applied": warm_n or warm_a,
        "seconds_numba": round(t_numba, 6),
        "seconds_resolved": round(t_resolved, 6),
        "resolved_backend": resolved,
        "openmp_linked": omp,
        "ratio_resolved_over_numba": round(t_resolved / t_numba, 4) if t_numba > 0 else None,
        "speedup_numba_over_resolved": round(t_numba / t_resolved, 4) if t_resolved > 0 else None,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
