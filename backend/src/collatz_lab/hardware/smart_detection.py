"""
Structured, JSON-safe metadata for automatic hardware probes.

Used by the API so dashboards and future tooling can reason about *how* a row
was produced and *what* to build next when Collatz cannot execute on a device yet.
"""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = 1

# Ordered research targets for non-CUDA Collatz GPU work (documentation / planning).
BACKEND_RESEARCH_BY_VENDOR: dict[str, list[str]] = {
    "nvidia": ["cuda_numba"],
    "apple": ["metal", "mlx", "mps"],
    "amd": ["rocm", "hip", "vulkan_compute"],
    "intel": ["level_zero", "sycl", "oneapi"],
    "unknown": ["vendor_specific_compute"],
}


def _block(
    *,
    probes: list[str],
    collatz_gpu_executable: bool,
    primary_signal: str,
    vendor: str,
    notes: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "probes": probes,
        "collatz_gpu_executable": collatz_gpu_executable,
        "primary_signal": primary_signal,
        "vendor_class": vendor,
        "next_backends_research": list(BACKEND_RESEARCH_BY_VENDOR.get(vendor, BACKEND_RESEARCH_BY_VENDOR["unknown"])),
    }
    if notes:
        payload["development_note"] = notes
    if extra:
        payload["signals"] = extra
    return {"smart_detection": payload}


def cpu_smart_detection(
    *,
    probes: list[str],
    usage_probe: str | None,
    extra_signals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sig = extra_signals.copy() if extra_signals else {}
    if usage_probe:
        sig["cpu_usage_probe"] = usage_probe
    return {
        "smart_detection": {
            "schema_version": SCHEMA_VERSION,
            "probes": probes,
            "collatz_cpu_executable": True,
            "primary_signal": "python_os_cpu",
            "vendor_class": "cpu",
            "next_backends_research": ["numba_cpu_jit", "threadpool_tuning"],
            **({"signals": sig} if sig else {}),
        }
    }


def nvidia_smi_smart_detection(
    *,
    runtime_ready: bool,
    driver_version: str,
    smi_index: str,
) -> dict[str, Any]:
    return _block(
        probes=["nvidia-smi_query_gpu", "nvidia-smi_utilization_optional", "numba_cuda_probe"],
        collatz_gpu_executable=runtime_ready,
        primary_signal="nvidia-smi",
        vendor="nvidia",
        notes=None if runtime_ready else "Driver reported GPU; Numba/CUDA runtime may still be incomplete.",
        extra={"driver_version": driver_version, "smi_index": smi_index},
    )


def display_probe_smart_detection(
    *,
    vendor: str,
    probe_tool: str,
    collatz_gpu_executable: bool = False,
    raw_snippet: str | None = None,
) -> dict[str, Any]:
    note = (
        "No Collatz GPU kernel targets this adapter class yet; use this JSON for backend spikes "
        "(Metal / ROCm / SYCL, etc.)."
    )
    extra: dict[str, Any] = {"probe_tool": probe_tool}
    if raw_snippet:
        extra["raw_snippet"] = raw_snippet[:500]
    return _block(
        probes=[probe_tool, "platform_context_implicit"],
        collatz_gpu_executable=collatz_gpu_executable,
        primary_signal=probe_tool,
        vendor=vendor,
        notes=note,
        extra=extra,
    )
