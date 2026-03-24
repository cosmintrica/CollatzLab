#!/usr/bin/env python3
"""
Local verification: hardware snapshot, CPU kernels, and GPU (CUDA or Apple MPS).

Run from repo root:
  PYTHONPATH=backend/src python3 scripts/verify_native_compute.py

Or with venv:
  .venv/bin/python scripts/verify_native_compute.py
  (ensure cwd is repo root so backend/src is found)
"""
from __future__ import annotations

import os
import sys


def _bootstrap_path() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(root, "backend", "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


def main() -> int:
    _bootstrap_path()

    from collatz_lab.hardware import discover_hardware, gpu_execution_ready
    from collatz_lab.hardware.gpu import cuda_gpu_execution_ready

    try:
        from collatz_lab import mps_collatz
    except Exception as exc:  # pragma: no cover
        print(f"mps_collatz import failed: {exc}")
        mps_collatz = None

    from collatz_lab.services import (
        GPU_KERNEL,
        GPU_SIEVE_KERNEL,
        compute_range_metrics,
        metrics_accelerated,
        metrics_direct,
    )

    print("== Hardware capabilities (discover_hardware) ==")
    for cap in discover_hardware():
        kh = ",".join(cap.supported_kernels or []) or "(none)"
        print(f"  [{cap.kind}] {cap.slug}: {cap.label!r} -> {kh}")

    errs = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal errs
        if cond:
            print(f"  OK {name}")
        else:
            print(f"  FAIL {name} {detail}")
            errs += 1

    print("\n== CPU: cpu-direct vs cpu-parallel [1..5000] ==")
    d = compute_range_metrics(1, 5000, kernel="cpu-direct")
    p = compute_range_metrics(1, 5000, kernel="cpu-parallel")
    check(
        "max_total_stopping_time",
        d.max_total_stopping_time == p.max_total_stopping_time,
        f"{d.max_total_stopping_time} vs {p.max_total_stopping_time}",
    )
    check(
        "max_stopping_time",
        d.max_stopping_time == p.max_stopping_time,
        f"{d.max_stopping_time} vs {p.max_stopping_time}",
    )
    check("max_excursion", d.max_excursion == p.max_excursion, f"{d.max_excursion} vs {p.max_excursion}")
    check("processed", d.processed == p.processed, f"{d.processed} vs {p.processed}")

    print("\n== CPU: cpu-sieve vs cpu-parallel-odd [1..3000] ==")
    s = compute_range_metrics(1, 3000, kernel="cpu-sieve")
    po = compute_range_metrics(1, 3000, kernel="cpu-parallel-odd")
    check("sieve vs parallel-odd processed count", s.processed == po.processed, f"{s.processed} vs {po.processed}")
    check(
        "sieve vs parallel-odd max_stopping_time",
        s.max_stopping_time == po.max_stopping_time,
        f"{s.max_stopping_time} vs {po.max_stopping_time}",
    )
    check(
        "sieve vs parallel-odd max_excursion",
        s.max_excursion == po.max_excursion,
        f"{s.max_excursion} vs {po.max_excursion}",
    )
    print(
        "  (note) max_total_stopping_time may differ: sieve uses table-assisted early termination "
        "vs stepwise odd descent — both kernels are covered by dedicated pytest suites."
    )

    print("\n== CPU: spot-check metrics_direct vs accelerated (n=27, 9663) ==")
    for n in (27, 9663):
        md, ma = metrics_direct(n), metrics_accelerated(n)
        check(f"n={n} direct==accelerated", md == ma, f"{md} vs {ma}")

    cuda_ok = cuda_gpu_execution_ready()
    mps_ok = bool(mps_collatz and mps_collatz.mps_accelerated_available())
    print(f"\n== GPU backends: CUDA(Numba)={cuda_ok}  MPS(PyTorch)={mps_ok}  gpu_execution_ready={gpu_execution_ready()} ==")

    if gpu_execution_ready():
        print("\n== GPU: gpu-collatz-accelerated vs cpu-direct [1..5000] ==")
        g = compute_range_metrics(1, 5000, kernel=GPU_KERNEL)
        check("gpu max_total", g.max_total_stopping_time == d.max_total_stopping_time)
        check("gpu max_stopping", g.max_stopping_time == d.max_stopping_time)
        check("gpu max_excursion", g.max_excursion == d.max_excursion)

        print("\n== GPU: gpu-sieve vs cpu-sieve [1..3000] ==")
        gs = compute_range_metrics(1, 3000, kernel=GPU_SIEVE_KERNEL)
        cs = compute_range_metrics(1, 3000, kernel="cpu-sieve")
        check("gpu-sieve max_total", gs.max_total_stopping_time == cs.max_total_stopping_time)
        check("gpu-sieve max_stopping", gs.max_stopping_time == cs.max_stopping_time)
        check("gpu-sieve max_excursion", gs.max_excursion == cs.max_excursion)

        print("\n== GPU: single-seed vs metrics_direct ==")
        for n in (1, 2, 27, 9663, 1000):
            gg = compute_range_metrics(n, n, kernel=GPU_KERNEL)
            ref = metrics_direct(n)
            check(
                f"seed {n} total",
                gg.max_total_stopping_time["value"] == ref.total_stopping_time,
            )
            check(
                f"seed {n} stopping",
                gg.max_stopping_time["value"] == ref.stopping_time,
            )
            check(
                f"seed {n} excursion",
                gg.max_excursion["value"] == ref.max_excursion,
            )
    else:
        print("\n== GPU checks skipped ==")
        print("  Install PyTorch MPS on Apple Silicon:  pip install -e 'backend[dev,mps]'")
        print("  Or NVIDIA CUDA stack:  pip install -e 'backend[dev,gpu]'")

    if errs:
        print(f"\nFAILED with {errs} check(s).")
        return 1
    print("\nAll automated checks passed.")
    print("  Throughput sweep (cpu-sieve vs gpu-sieve, MPS presets): scripts/benchmark_mac_throughput.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
