"""
Hardware discovery, CUDA shim (Windows), and kernel routing metadata.

Public API is stable — import from ``collatz_lab.hardware`` as before.
"""

from .constants import (
    CPU_ACCELERATED_KERNEL,
    CPU_BARINA_KERNEL,
    CPU_DIRECT_KERNEL,
    CPU_KERNELS,
    CPU_PARALLEL_KERNEL,
    CPU_PARALLEL_ODD_KERNEL,
    CPU_SIEVE_KERNEL,
    GPU_KERNEL,
    GPU_SIEVE_KERNEL,
)
from .discovery import discover_hardware
from .gpu import cuda_gpu_execution_ready, gpu_execution_diagnostics, gpu_execution_ready
from .selection import select_worker_execution_profile, validate_execution_request

__all__ = [
    "CPU_ACCELERATED_KERNEL",
    "CPU_BARINA_KERNEL",
    "CPU_DIRECT_KERNEL",
    "CPU_KERNELS",
    "CPU_PARALLEL_KERNEL",
    "CPU_PARALLEL_ODD_KERNEL",
    "CPU_SIEVE_KERNEL",
    "GPU_KERNEL",
    "GPU_SIEVE_KERNEL",
    "discover_hardware",
    "cuda_gpu_execution_ready",
    "gpu_execution_diagnostics",
    "gpu_execution_ready",
    "select_worker_execution_profile",
    "validate_execution_request",
]
