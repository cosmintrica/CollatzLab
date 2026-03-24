"""Compile native C sieve and check aggregate JSON vs Numba (optional; needs ``cc``)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not shutil.which("cc"), reason="no C compiler (cc) on PATH")


def test_native_c_aggregate_matches_cpu_sieve():
    root = Path(__file__).resolve().parents[2]
    kit = root / "scripts" / "native_sieve_kit"
    src = kit / "sieve_descent.c"
    out = kit / "sieve_descent"
    subprocess.run(
        ["cc", "-std=c11", "-O3", "-o", str(out), str(src)],
        check=True,
    )

    first_odd = 1
    odd_count = 5_000
    last_linear = first_odd + 2 * (odd_count - 1)

    p = subprocess.run(
        [str(out), "verify", str(first_odd), str(odd_count)],
        cwd=str(kit),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("{")), "")
    assert line
    c_agg = json.loads(line)

    # Import after optional heavy path
    sys.path.insert(0, str(root / "backend" / "src"))
    from collatz_lab.services import compute_range_metrics_sieve_odd

    numba = compute_range_metrics_sieve_odd(1, last_linear)

    assert c_agg["processed"] == numba.processed
    assert c_agg["last_processed"] == numba.last_processed
    assert c_agg["max_total_stopping_time"] == numba.max_total_stopping_time
    assert c_agg["max_stopping_time"] == numba.max_stopping_time
    assert c_agg["max_excursion"] == numba.max_excursion
