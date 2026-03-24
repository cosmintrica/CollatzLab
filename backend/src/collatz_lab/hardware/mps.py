"""Apple Metal Performance Shaders (MPS) backend probe via PyTorch."""

from __future__ import annotations

import platform
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def mps_execution_diagnostics() -> dict[str, Any]:
    if platform.system() != "Darwin":
        return {
            "ready": False,
            "reason": "MPS is only available on macOS.",
            "backend": "mps",
        }
    try:
        import torch
    except ImportError:
        return {
            "ready": False,
            "reason": "PyTorch is not installed. Install backend[mps] for Apple GPU compute.",
            "backend": "mps",
        }
    if not torch.backends.mps.is_built():
        return {
            "ready": False,
            "reason": "This PyTorch build does not include MPS.",
            "backend": "mps",
        }
    if not torch.backends.mps.is_available():
        return {
            "ready": False,
            "reason": "MPS is not available (headless VM, older OS, or GPU unavailable).",
            "backend": "mps",
        }
    return {
        "ready": True,
        "reason": "PyTorch MPS is available for Collatz GPU kernels.",
        "backend": "mps",
    }


def mps_execution_ready() -> bool:
    return bool(mps_execution_diagnostics()["ready"])
