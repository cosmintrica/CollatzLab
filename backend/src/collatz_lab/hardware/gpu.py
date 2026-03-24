"""Numba CUDA probe and readiness (after Windows shim)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from .mps import mps_execution_ready as _mps_execution_ready
from .windows_cuda import ensure_cuda_shim

ensure_cuda_shim()

try:
    from numba import cuda
except Exception:  # pragma: no cover - optional dependency
    cuda = None


@lru_cache(maxsize=1)
def gpu_execution_diagnostics() -> dict[str, object]:
    shim_root = Path(os.getenv("CUDA_HOME", "")) if os.getenv("CUDA_HOME") else None
    if cuda is None:
        return {
            "ready": False,
            "reason": "Numba CUDA is not installed.",
            "cuda_home": str(shim_root) if shim_root else None,
        }

    try:
        ready = bool(cuda.is_available())
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "ready": False,
            "reason": f"CUDA runtime probe failed: {exc}",
            "cuda_home": str(shim_root) if shim_root else None,
        }

    device_count = 0
    device_label = None
    device_error = ""
    try:
        devices = list(cuda.gpus)
        device_count = len(devices)
        if devices:
            device = devices[0]
            name = getattr(device, "name", None)
            device_label = name.decode() if hasattr(name, "decode") else name
    except Exception as exc:  # pragma: no cover - defensive
        device_error = str(exc)

    if ready:
        reason = "CUDA runtime and NVVM are available."
    elif device_count > 0:
        reason = (
            "GPU is visible to the driver, but CUDA runtime compilation is not ready. "
            f"{device_error}".strip()
        )
    else:
        reason = "No executable CUDA runtime was detected."

    return {
        "ready": ready,
        "reason": reason,
        "device_count": device_count,
        "device_label": device_label,
        "cuda_home": str(shim_root) if shim_root else None,
    }


def cuda_gpu_execution_ready() -> bool:
    """True when Numba CUDA can compile and run kernels (NVIDIA path)."""
    return bool(gpu_execution_diagnostics()["ready"])


def gpu_execution_ready() -> bool:
    """True when any Collatz GPU backend can run (CUDA or Apple MPS)."""
    return cuda_gpu_execution_ready() or _mps_execution_ready()
