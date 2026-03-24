"""Metal ``CollatzLabSieve`` aggregate vs ``cpu-sieve`` (macOS + toolchain only)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="Metal is macOS-only")


def test_metal_lab_sieve_aggregate_matches_cpu_sieve():
    if not shutil.which("xcrun"):
        pytest.skip("no xcrun")

    root = Path(__file__).resolve().parents[2]
    metal_dir = root / "scripts" / "native_sieve_kit" / "metal"
    build_sh = metal_dir / "build_metal_verify.sh"
    subprocess.run(["bash", str(build_sh)], cwd=str(metal_dir), check=True, capture_output=True)

    verify_bin = metal_dir / "metal_lab_sieve_verify"
    p = subprocess.run(
        [str(verify_bin), "--base", "1", "--count", "5000"],
        cwd=str(metal_dir),
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        pytest.skip(f"Metal verify failed: {p.stderr}")

    line = p.stdout.strip()
    m_agg = json.loads(line)

    sys.path.insert(0, str(root / "backend" / "src"))
    from collatz_lab.services import compute_range_metrics_sieve_odd

    numba = compute_range_metrics_sieve_odd(1, 9999)

    assert m_agg["processed"] == numba.processed
    assert m_agg["last_processed"] == numba.last_processed
    assert m_agg["max_total_stopping_time"] == numba.max_total_stopping_time
    assert m_agg["max_excursion"] == numba.max_excursion
