"""NVIDIA driver / nvidia-smi discovery for GPU capabilities."""

from __future__ import annotations

from ..schemas import HardwareCapability
from .constants import GPU_KERNEL, GPU_SIEVE_KERNEL
from .gpu import cuda_gpu_execution_ready, gpu_execution_diagnostics
from .smart_detection import nvidia_smi_smart_detection
from .util import parse_float, run_command


def _nvidia_runtime_metrics() -> dict[str, dict[str, float]]:
    output = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return {}

    metrics: dict[str, dict[str, float]] = {}
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        index, gpu_usage, memory_used, memory_total = parts
        gpu_value = parse_float(gpu_usage)
        used_value = parse_float(memory_used)
        total_value = parse_float(memory_total)
        payload: dict[str, float] = {}
        if gpu_value is not None:
            payload["usage_percent"] = gpu_value
        if used_value is not None:
            payload["memory_used_mib"] = used_value
        if total_value is not None:
            payload["memory_total_mib"] = total_value
        if used_value is not None and total_value not in (None, 0):
            payload["memory_usage_percent"] = round((used_value / total_value) * 100.0, 2)
        metrics[index] = payload
    return metrics


def detect_nvidia_gpus() -> list[HardwareCapability]:
    output = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return []

    diagnostics = gpu_execution_diagnostics()
    runtime_ready = cuda_gpu_execution_ready()
    runtime_metrics = _nvidia_runtime_metrics()
    capabilities: list[HardwareCapability] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        index, name, memory_mib, driver_version = parts
        live_metrics = runtime_metrics.get(index, {})
        capabilities.append(
            HardwareCapability(
                kind="gpu",
                slug=f"gpu-nvidia-{index}",
                label=name,
                available=True,
                supported_hardware=["gpu"],
                supported_kernels=[GPU_KERNEL, GPU_SIEVE_KERNEL] if runtime_ready else [],
                metadata={
                    "vendor": "nvidia",
                    "index": index,
                    "memory_mib": int(memory_mib),
                    "driver_version": driver_version,
                    "execution_ready": runtime_ready,
                    "diagnostic": diagnostics["reason"],
                    "cuda_home": diagnostics["cuda_home"],
                    **live_metrics,
                    **nvidia_smi_smart_detection(
                        runtime_ready=runtime_ready,
                        driver_version=driver_version,
                        smi_index=index,
                    ),
                },
            )
        )
    return capabilities
