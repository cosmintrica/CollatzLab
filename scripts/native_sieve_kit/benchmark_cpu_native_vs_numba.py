#!/usr/bin/env python3
"""
Benchmark **native C** odd-sieve (``sieve_descent bench``) vs **Numba cpu-sieve** on the same odd count.

Apple Silicon (or any host with ``cc``): compares wall time only; **correctness** is enforced elsewhere
(``compare_native_cpu_vs_lab.py``, pytest).

  cd CollatzLab
  PYTHONPATH=backend/src python3 scripts/native_sieve_kit/benchmark_cpu_native_vs_numba.py [--odd-count 500000]

Does **not** use Metal. Safe on Linux/Windows if ``cc`` and C source exist.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--odd-count", type=int, default=500_000)
    args = ap.parse_args()
    oc = max(1, args.odd_count)
    first_odd = 1
    last_linear = first_odd + 2 * (oc - 1)

    root = _repo_root()
    kit = os.path.join(root, "scripts", "native_sieve_kit")
    bin_path = os.path.join(kit, "sieve_descent")
    src = os.path.join(kit, "sieve_descent.c")
    if not os.path.isfile(src):
        print("missing sieve_descent.c", file=sys.stderr)
        return 2
    if not os.path.isfile(bin_path):
        subprocess.run(["cc", "-std=c11", "-O3", "-o", bin_path, src], cwd=kit, check=True)

    t0 = time.perf_counter()
    subprocess.run([bin_path, "bench", str(first_odd), str(oc)], cwd=kit, check=True, capture_output=True)
    c_wall = time.perf_counter() - t0

    sys.path.insert(0, os.path.join(root, "backend", "src"))
    from collatz_lab.services import compute_range_metrics_sieve_odd

    t1 = time.perf_counter()
    compute_range_metrics_sieve_odd(1, last_linear)
    py_wall = time.perf_counter() - t1

    ratio = py_wall / c_wall if c_wall > 0 else 0.0
    print(
        json.dumps(
            {
                "odd_count": oc,
                "linear_range": [1, last_linear],
                "native_c_wall_s": round(c_wall, 4),
                "numba_cpu_sieve_wall_s": round(py_wall, 4),
                "numba_vs_native_c_ratio": round(ratio, 2),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
