"""
Differential test – level 3: same interval, independent backends, same semantics.

WHY THIS EXISTS
===============
``validate_run`` checks aggregate windows using ``_compute_descent_odd_only_reference``,
which delegates to ``compute_range_metrics_parallel_descent_odd`` — another Numba kernel.
That is *Numba-vs-Numba*: a consistency / reproducibility check, not a cross-check against
the source of truth.

The only per-seed SoT cross-check (Python bigint ``metrics_descent_direct``) is in
``_validate_record_seeds``, but only for record-breaking seeds.  A run with no new records
gets zero bigint validation of its window aggregates.

This test closes that gap: the Python ``sieve_reference`` is the SoT anchor, and every
fast backend must agree with it on the same small interval.

VALIDATION ASYMMETRY (documented here to match the in-code docstring)
======================================================================
  validate_run windows  (cpu-sieve):        Numba descent-odd  vs Numba sieve    [consistency]
  validate_run windows  (cpu-parallel-odd): Python bigint      vs Numba parallel [SoT cross-check]
  validate_run record seeds (all kernels):  Python bigint      vs any kernel     [SoT cross-check]
  THIS TEST:                                Python sieve_ref   vs all fast paths [systematic-bug net]

SAFE INTERVALS
==============
[3, 999] and [1, 9999] — well below the int64 overflow threshold for any of these seeds.
``sieve_reference`` does not implement the overflow patch, so intervals must stay safe.
Seeds needing the overflow patch (very large n, e.g. >10^17) are NOT tested here; they are
covered by ``_validate_record_seeds`` via ``metrics_descent_direct`` (Python bigint).
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from collatz_lab.cpu_sieve_native_runtime import native_cpu_sieve_lib_path
from collatz_lab.hardware import gpu_execution_ready
from collatz_lab.services import compute_range_metrics, compute_range_metrics_sieve_odd
from collatz_lab.sieve_reference import odd_sieve_descent_linear_range


def _native_c_run_in_subprocess(start: int, end: int) -> dict | None:
    """Run ``compute_range_metrics_sieve_odd_native`` in a fresh subprocess.

    **Why subprocess:** on macOS + Apple Silicon, calling the native C ctypes
    path from within the same process that has both Numba (Homebrew libomp) and
    PyTorch (bundled libomp) loaded causes a SIGABRT from OpenMP's dual-instance
    detection (``__kmp_register_library_startup``).  This conflict is
    non-deterministic and impossible to probe in advance — it depends on the exact
    initialization order of OpenMP in the calling process.  Running native C in a
    fresh subprocess avoids the conflict and lets the test compare results safely.

    Returns a normalized dict if the subprocess succeeded, or ``None`` if it crashed
    or the library is not built (caller should ``pytest.skip``).
    """
    script = (
        "import json; "
        "from collatz_lab.cpu_sieve_native_runtime import compute_range_metrics_sieve_odd_native; "
        f"r = compute_range_metrics_sieve_odd_native({start}, {end}); "
        "print(json.dumps({"
        "'processed': r.processed,"
        "'last_processed': r.last_processed,"
        "'max_total_stopping_time': r.max_total_stopping_time,"
        "'max_stopping_time': r.max_stopping_time,"
        "'max_excursion': r.max_excursion"
        "}))"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        return json.loads(result.stdout.strip())
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Intervals that are safe for sieve_reference (no int64 overflow in orbit).
# Keep small so the test runs fast on CI without GPU/native.
# ---------------------------------------------------------------------------
SAFE_INTERVALS = [(3, 999), (1, 9_999)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(result) -> dict:
    """Return a dict of the five comparable aggregate fields.

    Accepts either a ``sieve_reference`` dict or an ``AggregateMetrics`` object.
    """
    if isinstance(result, dict):
        return {
            "processed": result["processed"],
            "last_processed": result["last_processed"],
            "max_total_stopping_time": result["max_total_stopping_time"],
            "max_stopping_time": result["max_stopping_time"],
            "max_excursion": result["max_excursion"],
        }
    return {
        "processed": result.processed,
        "last_processed": result.last_processed,
        "max_total_stopping_time": result.max_total_stopping_time,
        "max_stopping_time": result.max_stopping_time,
        "max_excursion": result.max_excursion,
    }


def _assert_matches_ref(label: str, ref: dict, candidate, start: int, end: int) -> None:
    cand = _normalize(candidate)
    assert cand["processed"] == ref["processed"], (
        f"[{label}] [{start},{end}] processed: got {cand['processed']}, ref {ref['processed']}"
    )
    assert cand["last_processed"] == ref["last_processed"], (
        f"[{label}] [{start},{end}] last_processed: got {cand['last_processed']}, ref {ref['last_processed']}"
    )
    assert cand["max_total_stopping_time"] == ref["max_total_stopping_time"], (
        f"[{label}] [{start},{end}] max_total_stopping_time mismatch:\n"
        f"  candidate: {cand['max_total_stopping_time']}\n"
        f"  reference: {ref['max_total_stopping_time']}"
    )
    assert cand["max_stopping_time"] == ref["max_stopping_time"], (
        f"[{label}] [{start},{end}] max_stopping_time mismatch:\n"
        f"  candidate: {cand['max_stopping_time']}\n"
        f"  reference: {ref['max_stopping_time']}"
    )
    assert cand["max_excursion"] == ref["max_excursion"], (
        f"[{label}] [{start},{end}] max_excursion mismatch:\n"
        f"  candidate: {cand['max_excursion']}\n"
        f"  reference: {ref['max_excursion']}"
    )


# ---------------------------------------------------------------------------
# Path 1: Python sieve_reference (always — this is the SoT anchor)
# Just verify it runs without error; it's used as the reference in all tests below.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("start,end", SAFE_INTERVALS)
def test_python_ref_runs(start: int, end: int) -> None:
    """sieve_reference must complete and return a valid aggregate."""
    ref = odd_sieve_descent_linear_range(start, end)
    assert ref["processed"] > 0
    assert ref["max_total_stopping_time"]["value"] > 0
    assert ref["max_excursion"]["value"] >= ref["max_total_stopping_time"]["n"]


# ---------------------------------------------------------------------------
# Path 2: Numba cpu-sieve  (always; forced via COLLATZ_CPU_SIEVE_BACKEND=numba)
#
# Even if the native C library is present, monkeypatch forces Numba so we
# specifically exercise the Numba JIT path, not native C.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("start,end", SAFE_INTERVALS)
def test_numba_sieve_matches_python_ref(monkeypatch, start: int, end: int) -> None:
    """Numba cpu-sieve vs Python sieve_reference (level-3 SoT cross-check).

    COLLATZ_CPU_SIEVE_BACKEND=numba forces Numba even when native C is present.
    ``cpu_sieve_resolve_backend`` short-circuits at mode=="numba" before checking
    ``native_cpu_sieve_available()``, so the lru_cache on the latter is irrelevant.
    """
    monkeypatch.setenv("COLLATZ_CPU_SIEVE_BACKEND", "numba")
    ref = odd_sieve_descent_linear_range(start, end)
    numba_result = compute_range_metrics_sieve_odd(start, end)
    _assert_matches_ref("numba cpu-sieve", ref, numba_result, start, end)


# ---------------------------------------------------------------------------
# Path 3: Native C shared library
# Skip if libsieve_descent_native.{dylib|so|dll} is not built.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    native_cpu_sieve_lib_path() is None,
    reason="Native C cpu-sieve library not built (run build_native_cpu_sieve_lib.sh)",
)
@pytest.mark.parametrize("start,end", SAFE_INTERVALS)
def test_native_c_sieve_matches_python_ref(start: int, end: int) -> None:
    """Native C cpu-sieve vs Python sieve_reference (level-3 SoT cross-check).

    Runs native C in a subprocess to avoid OpenMP dual-instance SIGABRT on macOS
    (Homebrew libomp from the native dylib vs PyTorch's bundled libomp in the same process).
    """
    native_result = _native_c_run_in_subprocess(start, end)
    if native_result is None:
        pytest.skip("Native C subprocess returned no result (library missing or libomp conflict)")
    ref = odd_sieve_descent_linear_range(start, end)
    _assert_matches_ref("native C cpu-sieve", ref, native_result, start, end)


# ---------------------------------------------------------------------------
# Path 4: GPU sieve (Metal/MPS or CUDA)
# Skip if no GPU backend is ready.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not gpu_execution_ready(),
    reason="No GPU backend available (Metal/MPS or CUDA required)",
)
@pytest.mark.parametrize("start,end", SAFE_INTERVALS)
def test_gpu_sieve_matches_python_ref(start: int, end: int) -> None:
    """GPU sieve (Metal/MPS or CUDA) vs Python sieve_reference (level-3 SoT cross-check)."""
    ref = odd_sieve_descent_linear_range(start, end)
    gpu_result = compute_range_metrics(start, end, kernel="gpu-sieve")
    _assert_matches_ref("gpu-sieve", ref, gpu_result, start, end)


# ---------------------------------------------------------------------------
# Path 3 vs Path 2: native C vs Numba (parity, when both available)
# This catches drift between the two fast CPU implementations independent of the ref.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    native_cpu_sieve_lib_path() is None,
    reason="Native C cpu-sieve library not built (run build_native_cpu_sieve_lib.sh)",
)
@pytest.mark.parametrize("start,end", SAFE_INTERVALS)
def test_native_c_vs_numba_parity(monkeypatch, start: int, end: int) -> None:
    """Native C and Numba must produce identical aggregates (regression guard).

    Runs native C in a subprocess (see ``_native_c_run_in_subprocess`` docstring for why).
    Numba is forced via monkeypatch so both paths are exercised independently.
    """
    native_result = _native_c_run_in_subprocess(start, end)
    if native_result is None:
        pytest.skip("Native C subprocess returned no result (library missing or libomp conflict)")
    monkeypatch.setenv("COLLATZ_CPU_SIEVE_BACKEND", "numba")
    numba_result = compute_range_metrics_sieve_odd(start, end)
    _assert_matches_ref("native-C vs Numba", _normalize(numba_result), native_result, start, end)
