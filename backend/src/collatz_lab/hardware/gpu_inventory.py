"""
GPU capability aggregation: NVIDIA SMI (executable path) + OS display probes (dev signal).

Deduplicates redundant NVIDIA rows from PCI/WMI when nvidia-smi already owns the device.
"""

from __future__ import annotations

import os

from ..schemas import HardwareCapability
from .adapters.display import probe_display_adapters
from .constants import GPU_KERNEL, GPU_SIEVE_KERNEL
from .mps import mps_execution_diagnostics, mps_execution_ready
from .nvidia import detect_nvidia_gpus


def _display_row_redundant_with_nvidia(cap: HardwareCapability, nvidia_caps: list[HardwareCapability]) -> bool:
    """Drop duplicate NVIDIA enumeration from lspci / WMI when SMI already returned GPUs."""
    if not nvidia_caps:
        return False
    vendor = (cap.metadata or {}).get("vendor")
    label = (cap.label or "").lower()
    if vendor == "nvidia":
        return True
    if "nvidia" in label:
        return True
    return False


def _upgrade_apple_gpu_for_mps(caps: list[HardwareCapability]) -> list[HardwareCapability]:
    if not mps_execution_ready():
        return caps
    diag = mps_execution_diagnostics()
    out: list[HardwareCapability] = []
    for cap in caps:
        if cap.kind != "gpu":
            out.append(cap)
            continue
        meta = dict(cap.metadata or {})
        vendor = str(meta.get("vendor") or "").lower()
        label_low = (cap.label or "").lower()
        is_apple_class = vendor == "apple" or "apple" in label_low
        if is_apple_class and not cap.supported_kernels:
            new_meta = {
                **meta,
                "execution_ready": True,
                "collatz_gpu_backend": "mps",
                "mps": {k: v for k, v in diag.items() if k in ("ready", "reason", "backend")},
                # No nvidia-smi on macOS; live GPU % is not exposed to user-space without root tools.
                "gpu_usage_probe": "unavailable_macos",
                "gpu_usage_hint": (
                    "macOS does not expose GPU utilization to this API. "
                    "Open Activity Monitor → CPU tab → View → Columns → enable “GPU %”, "
                    "then check the Python worker process while a GPU run is active."
                ),
            }
            new_meta["diagnostic"] = str(diag.get("reason", ""))
            sd = meta.get("smart_detection")
            if isinstance(sd, dict):
                new_meta["smart_detection"] = {**sd, "collatz_gpu_executable": True}
            out.append(
                cap.model_copy(
                    update={
                        "supported_hardware": ["gpu"],
                        "supported_kernels": [GPU_KERNEL, GPU_SIEVE_KERNEL],
                        "metadata": new_meta,
                    }
                )
            )
        else:
            out.append(cap)
    return out


def _inject_apple_metal_sieve_without_torch(caps: list[HardwareCapability]) -> list[HardwareCapability]:
    """Optional: advertise gpu-sieve on Apple GPU when native Metal helper exists but MPS is not."""
    if os.getenv("COLLATZ_GPU_SIEVE_METAL_WITHOUT_TORCH", "").strip().lower() not in {
        "1",
        "true",
        "yes",
    }:
        return caps
    if mps_execution_ready():
        return caps
    try:
        from ..gpu_sieve_metal_runtime import native_metal_sieve_available
    except Exception:
        return caps
    if not native_metal_sieve_available():
        return caps

    out: list[HardwareCapability] = []
    for cap in caps:
        if cap.kind != "gpu":
            out.append(cap)
            continue
        meta = dict(cap.metadata or {})
        vendor = str(meta.get("vendor") or "").lower()
        label_low = (cap.label or "").lower()
        is_apple = vendor == "apple" or "apple" in label_low
        kernels = list(cap.supported_kernels or [])
        if is_apple and GPU_SIEVE_KERNEL not in kernels:
            kernels.append(GPU_SIEVE_KERNEL)
            new_meta = {
                **meta,
                "execution_ready": True,
                "collatz_gpu_backend": "metal_native",
                "metal_native_note": (
                    "gpu-sieve via metal_sieve_chunk; COLLATZ_GPU_SIEVE_METAL_WITHOUT_TORCH=1. "
                    "gpu-collatz-accelerated still requires PyTorch MPS/CUDA."
                ),
            }
            out.append(
                cap.model_copy(
                    update={
                        "available": True,
                        "supported_hardware": ["gpu"],
                        "supported_kernels": kernels,
                        "metadata": new_meta,
                    }
                )
            )
        else:
            out.append(cap)
    return out


def collect_gpu_capabilities() -> list[HardwareCapability]:
    """
    Smart automatic inventory:
    1. Query NVIDIA driver / runtime via nvidia-smi (kernels populated when CUDA is ready).
    2. Always run OS display probes for additional adapters (iGPU, Apple GPU, etc.).
    3. Merge and dedupe so hybrid laptops still show Intel/AMD alongside NVIDIA.
    """
    nvidia_caps = detect_nvidia_gpus()
    display_caps = probe_display_adapters()
    extra = [c for c in display_caps if not _display_row_redundant_with_nvidia(c, nvidia_caps)]
    merged = _upgrade_apple_gpu_for_mps([*nvidia_caps, *extra])
    return _inject_apple_metal_sieve_without_torch(merged)
