"""Parity: native C vs Numba on safe ranges (needs ``cc`` + shared lib)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.skipif(not shutil.which("cc"), reason="no C compiler (cc) on PATH"),
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="native cpu-sieve shared library build is Darwin/Linux only in-kit",
    ),
]


def _clear_native_caches() -> None:
    from collatz_lab.cpu_sieve_native_runtime import clear_cpu_sieve_native_runtime_caches

    clear_cpu_sieve_native_runtime_caches()


def _build_native_lib(kit: Path) -> None:
    subprocess.run(
        ["bash", str(kit / "build_native_cpu_sieve_lib.sh")],
        cwd=str(kit),
        check=True,
    )


def _agg_payload(agg) -> dict:
    from collatz_lab.validation import _aggregate_validation_payload

    return _aggregate_validation_payload(agg)


def test_native_cpu_sieve_matches_numba_on_ranges(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    kit = root / "scripts" / "native_sieve_kit"
    _build_native_lib(kit)
    monkeypatch.setenv("COLLATZ_LAB_ROOT", str(root))
    monkeypatch.delenv("COLLATZ_CPU_SIEVE_NATIVE_LIB", raising=False)

    from collatz_lab.services import compute_range_metrics_sieve_odd

    for start, end in ((1, 8_000), (10_001, 22_222), (500_001, 503_000)):
        _clear_native_caches()
        monkeypatch.setenv("COLLATZ_CPU_SIEVE_BACKEND", "numba")
        ref = compute_range_metrics_sieve_odd(start, end)

        _clear_native_caches()
        monkeypatch.setenv("COLLATZ_CPU_SIEVE_BACKEND", "native")
        nat = compute_range_metrics_sieve_odd(start, end)
        assert _agg_payload(ref) == _agg_payload(nat), f"range [{start},{end}]"


def test_native_cpu_sieve_checkpoint_resume_matches_numba(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    kit = root / "scripts" / "native_sieve_kit"
    _build_native_lib(kit)
    monkeypatch.setenv("COLLATZ_LAB_ROOT", str(root))
    start_at = 15_001
    end = 18_000

    from collatz_lab.services import compute_range_metrics_sieve_odd

    _clear_native_caches()
    monkeypatch.setenv("COLLATZ_CPU_SIEVE_BACKEND", "numba")
    ref = compute_range_metrics_sieve_odd(1, end, start_at=start_at)

    _clear_native_caches()
    monkeypatch.setenv("COLLATZ_CPU_SIEVE_BACKEND", "native")
    nat = compute_range_metrics_sieve_odd(1, end, start_at=start_at)
    assert _agg_payload(ref) == _agg_payload(nat)


def test_forced_native_without_lib_raises(monkeypatch):
    """No fallback: native mode requires the shared library (simulated missing)."""
    monkeypatch.setenv("COLLATZ_CPU_SIEVE_BACKEND", "native")
    monkeypatch.setattr(
        "collatz_lab.cpu_sieve_native_runtime.native_cpu_sieve_available",
        lambda: False,
    )
    _clear_native_caches()

    from collatz_lab.services import compute_range_metrics_sieve_odd

    with pytest.raises(ValueError, match="native"):
        compute_range_metrics_sieve_odd(1, 100)


def test_auto_resolve_native_when_lib_marked_available(monkeypatch):
    monkeypatch.delenv("COLLATZ_CPU_SIEVE_BACKEND", raising=False)
    monkeypatch.setattr(
        "collatz_lab.cpu_sieve_native_runtime.native_cpu_sieve_available",
        lambda: True,
    )
    from collatz_lab.cpu_sieve_native_runtime import cpu_sieve_resolve_backend

    assert cpu_sieve_resolve_backend() == "native"


def test_auto_resolve_numba_when_lib_unavailable(monkeypatch):
    monkeypatch.delenv("COLLATZ_CPU_SIEVE_BACKEND", raising=False)
    monkeypatch.setattr(
        "collatz_lab.cpu_sieve_native_runtime.native_cpu_sieve_available",
        lambda: False,
    )
    from collatz_lab.cpu_sieve_native_runtime import cpu_sieve_resolve_backend

    assert cpu_sieve_resolve_backend() == "numba"


def test_native_cpu_sieve_auto_matches_numba_when_built(monkeypatch):
    """With a real dylib/so, ``auto`` should use native and match Numba aggregates."""
    root = Path(__file__).resolve().parents[2]
    kit = root / "scripts" / "native_sieve_kit"
    _build_native_lib(kit)
    monkeypatch.setenv("COLLATZ_LAB_ROOT", str(root))
    monkeypatch.delenv("COLLATZ_CPU_SIEVE_NATIVE_LIB", raising=False)
    monkeypatch.delenv("COLLATZ_CPU_SIEVE_BACKEND", raising=False)

    from collatz_lab.services import compute_range_metrics_sieve_odd

    start, end = 1, 12_000
    _clear_native_caches()
    monkeypatch.setenv("COLLATZ_CPU_SIEVE_BACKEND", "numba")
    ref = compute_range_metrics_sieve_odd(start, end)

    _clear_native_caches()
    monkeypatch.setenv("COLLATZ_CPU_SIEVE_BACKEND", "auto")
    auto = compute_range_metrics_sieve_odd(start, end)
    assert _agg_payload(ref) == _agg_payload(auto)
