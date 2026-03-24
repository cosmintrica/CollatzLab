"""
Native **C** odd-only ``cpu-sieve`` backend (same loop as ``sieve_descent.c`` / Numba kernel).

- **Default:** ``COLLATZ_CPU_SIEVE_BACKEND=auto`` — uses the native ``.dylib`` / ``.so`` if present,
  otherwise Numba. Force ``numba`` or ``native`` for debugging.
- **Library:** ``libsieve_descent_native.{dylib|so}`` — build
  ``bash scripts/native_sieve_kit/build_native_cpu_sieve_lib.sh`` (OpenMP when libomp / ``-fopenmp``
  is available; else sequential).
- **Override path:** ``COLLATZ_CPU_SIEVE_NATIVE_LIB`` → absolute path to the shared library.

Overflow semantics match Numba: C marks ``-1``; Python applies ``metrics_descent_direct`` and
``_OverflowPatch`` exactly like ``compute_range_metrics_sieve_odd`` (Numba path).

**Threads:** OpenMP respects ``OMP_NUM_THREADS`` (and related env vars) when the library was built
with OpenMP; Numba path still uses ``set_num_threads`` from the compute profile.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _workspace_root() -> Path:
    return Path(os.getenv("COLLATZ_LAB_ROOT", Path.cwd())).resolve()


def _default_native_lib_paths() -> list[Path]:
    root = _workspace_root()
    pkg = Path(__file__).resolve().parent
    repo_via_file = pkg.parent.parent.parent
    suf = _shared_lib_suffix()
    if not suf:
        return []
    name = f"libsieve_descent_native.{suf}"
    out: list[Path] = []
    env = os.getenv("COLLATZ_CPU_SIEVE_NATIVE_LIB", "").strip()
    if env:
        out.append(Path(env).expanduser())
    out.append(root / "scripts" / "native_sieve_kit" / name)
    out.append(repo_via_file / "scripts" / "native_sieve_kit" / name)
    return out


def _shared_lib_suffix() -> str | None:
    system = platform.system()
    if system == "Darwin":
        return "dylib"
    if system == "Linux":
        return "so"
    if system == "Windows":
        return "dll"
    return None


@lru_cache(maxsize=1)
def native_cpu_sieve_lib_path() -> Path | None:
    for p in _default_native_lib_paths():
        if p.is_file():
            return p
    return None


@lru_cache(maxsize=1)
def _ctypes_lib() -> ctypes.CDLL:
    path = native_cpu_sieve_lib_path()
    if path is None:
        raise RuntimeError(
            "Native cpu-sieve library not found. Build: "
            "bash scripts/native_sieve_kit/build_native_cpu_sieve_lib.sh "
            "or set COLLATZ_CPU_SIEVE_NATIVE_LIB to the .dylib/.so path."
        )
    lib = ctypes.CDLL(str(path))
    fn = lib.collatz_lab_cpu_sieve_odd_fill
    fn.argtypes = [
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int64),
    ]
    fn.restype = None
    try:
        bi = lib.collatz_lab_cpu_sieve_build_info
        bi.argtypes = []
        bi.restype = ctypes.c_int32
    except AttributeError:
        bi = None
    setattr(lib, "_collatz_cpu_sieve_build_info", bi)
    return lib


def clear_cpu_sieve_native_runtime_caches() -> None:
    """Invalidate resolved library path and ctypes handle (tests / benchmarks after rebuild)."""
    native_cpu_sieve_lib_path.cache_clear()
    _ctypes_lib.cache_clear()


def native_cpu_sieve_openmp_linked() -> bool | None:
    """Whether the loaded dylib/so was built with OpenMP.

    ``True`` / ``False`` when ``collatz_lab_cpu_sieve_build_info`` exists; ``None`` if the symbol
    is missing (older builds) or the library cannot be loaded.
    """
    if native_cpu_sieve_lib_path() is None:
        return None
    try:
        lib = _ctypes_lib()
    except Exception:
        return None
    bi = getattr(lib, "_collatz_cpu_sieve_build_info", None)
    if bi is None:
        return None
    return int(bi()) != 0


def native_cpu_sieve_available() -> bool:
    if _shared_lib_suffix() is None:
        return False
    try:
        _ctypes_lib()
        return True
    except Exception:
        return False


def cpu_sieve_backend_mode() -> str:
    raw = os.getenv("COLLATZ_CPU_SIEVE_BACKEND", "auto").strip().lower()
    if raw in {"auto", "numba", "native"}:
        return raw
    return "auto"


def cpu_sieve_resolve_backend() -> str:
    """Effective backend for ``cpu-sieve`` fill: ``native`` or ``numba``.

    ``COLLATZ_CPU_SIEVE_BACKEND=native`` raises ``ValueError`` if the shared library is missing
    (no silent fallback).
    """
    mode = cpu_sieve_backend_mode()
    if mode == "numba":
        return "numba"
    if mode == "native":
        if not native_cpu_sieve_available():
            raise ValueError(
                "COLLATZ_CPU_SIEVE_BACKEND=native but libsieve_descent_native is missing. "
                "Build: bash scripts/native_sieve_kit/build_native_cpu_sieve_lib.sh"
            )
        return "native"
    return "native" if native_cpu_sieve_available() else "numba"


def diagnostics_native_cpu_sieve() -> dict[str, Any]:
    path = native_cpu_sieve_lib_path()
    resolved: str | None = None
    resolved_err: str | None = None
    try:
        resolved = cpu_sieve_resolve_backend()
    except ValueError as exc:
        resolved_err = str(exc)
    return {
        "platform": platform.system(),
        "backend_mode": cpu_sieve_backend_mode(),
        "resolved_backend": resolved,
        "resolved_error": resolved_err,
        "openmp_linked": native_cpu_sieve_openmp_linked(),
        "shared_lib_suffix": _shared_lib_suffix(),
        "library_path": str(path) if path else None,
        "available": native_cpu_sieve_available(),
        "would_use_native": resolved == "native",
        # Back-compat for clients that read the old key:
        "would_use_native_error": resolved_err,
    }


def compute_range_metrics_sieve_odd_native(
    start: int,
    end: int,
    *,
    start_at: int | None = None,
    sample_limit: int = 12,
    profile: Any = None,
) -> Any:
    """Odd-only cpu-sieve via C ``collatz_lab_cpu_sieve_odd_fill`` + Python overflow merge."""
    import numpy as np

    from .services import _cpu_sieve_odd_finalize_from_arrays

    if np is None:
        raise ValueError("numpy is required for native cpu-sieve.")
    if start < 1 or end < start:
        raise ValueError("Invalid run interval.")
    first = start_at if start_at is not None else start
    if first < start or first > end:
        raise ValueError("Checkpoint start is outside the run interval.")

    first_odd = first if first & 1 else first + 1
    if first_odd > end:
        from .services import AggregateMetrics

        return AggregateMetrics(
            processed=0,
            last_processed=end,
            max_total_stopping_time={"n": first, "value": 0},
            max_stopping_time={"n": first, "value": 0},
            max_excursion={"n": first, "value": 0},
            sample_records=[],
        )

    odd_count = ((end - first_odd) // 2) + 1
    lib = _ctypes_lib()
    total_steps = np.empty(odd_count, dtype=np.int32)
    stopping_steps = np.empty(odd_count, dtype=np.int32)
    max_excursions = np.empty(odd_count, dtype=np.int64)

    lib.collatz_lab_cpu_sieve_odd_fill(
        ctypes.c_int64(first_odd),
        ctypes.c_int32(int(odd_count)),
        total_steps.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        stopping_steps.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        max_excursions.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )

    return _cpu_sieve_odd_finalize_from_arrays(
        first_odd=first_odd,
        odd_count=int(odd_count),
        range_end=end,
        total_steps=total_steps,
        stopping_steps=stopping_steps,
        max_excursions=max_excursions,
        sample_limit=sample_limit,
    )
