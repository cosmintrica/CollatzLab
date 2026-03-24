#!/usr/bin/env python3
"""
Compare native C ``sieve_descent`` aggregates to ``cpu-sieve`` (Numba) and optional Python reference.

  cd CollatzLab
  PYTHONPATH=backend/src python3 scripts/native_sieve_kit/compare_native_cpu_vs_lab.py

Requires: built ``scripts/native_sieve_kit/sieve_descent`` (``bash scripts/native_sieve_kit/build_c.sh``).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    root = _repo_root()
    kit = os.path.join(root, "scripts", "native_sieve_kit")
    binary = os.path.join(kit, "sieve_descent")
    if not os.path.isfile(binary):
        src = os.path.join(kit, "sieve_descent.c")
        print(f"Building {binary} from {src} …", file=sys.stderr)
        subprocess.run(
            ["cc", "-std=c11", "-O3", "-o", binary, src],
            cwd=kit,
            check=True,
        )

    sys.path.insert(0, os.path.join(root, "backend", "src"))
    from collatz_lab.services import compute_range_metrics_sieve_odd
    from collatz_lab.sieve_reference import odd_sieve_descent_linear_range

    first_odd = 1
    odd_count = 5_000
    last_linear = first_odd + 2 * (odd_count - 1)

    p = subprocess.run(
        [binary, "verify", str(first_odd), str(odd_count)],
        cwd=kit,
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        print(p.stderr, file=sys.stderr)
        return p.returncode

    line = ""
    for ln in p.stdout.strip().splitlines():
        if ln.startswith("{") and ln.endswith("}"):
            line = ln
    if not line:
        print("no JSON from C", p.stdout, file=sys.stderr)
        return 3

    c_agg = json.loads(line)
    py_ref = odd_sieve_descent_linear_range(1, last_linear)
    numba = compute_range_metrics_sieve_odd(1, last_linear)

    def norm_from_metrics(m):
        return {
            "processed": m.processed,
            "last_processed": m.last_processed,
            "max_total_stopping_time": m.max_total_stopping_time,
            "max_stopping_time": m.max_stopping_time,
            "max_excursion": m.max_excursion,
        }

    def norm(d):
        return {k: d[k] for k in ("processed", "last_processed", "max_total_stopping_time", "max_stopping_time", "max_excursion")}

    print("== C native ==", json.dumps(c_agg))
    print("== Python ref ==", json.dumps(norm(py_ref)))
    print("== Numba cpu-sieve ==", json.dumps(norm_from_metrics(numba)))

    ok = c_agg == norm(py_ref) == norm_from_metrics(numba)
    print("parity:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
