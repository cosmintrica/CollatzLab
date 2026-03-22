from __future__ import annotations

import os
import platform
import re
import shutil
import site
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from .schemas import HardwareCapability


CPU_DIRECT_KERNEL = "cpu-direct"
CPU_ACCELERATED_KERNEL = "cpu-accelerated"
CPU_PARALLEL_KERNEL = "cpu-parallel"
CPU_PARALLEL_ODD_KERNEL = "cpu-parallel-odd"
CPU_KERNELS = [CPU_DIRECT_KERNEL, CPU_ACCELERATED_KERNEL, CPU_PARALLEL_KERNEL, CPU_PARALLEL_ODD_KERNEL]
GPU_KERNEL = "gpu-collatz-accelerated"
_DLL_DIRECTORY_HANDLES: list[object] = []


def _run_command(command: Iterable[str]) -> str | None:
    try:
        completed = subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _parse_float(value: str | None) -> float | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"-?\d+(?:[.,]\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", "."))
    except ValueError:
        return None


def _cpu_usage_percent() -> float | None:
    output = _run_command(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "[string]::Format([System.Globalization.CultureInfo]::InvariantCulture, '{0:F2}', (Get-Counter '\\Processor(_Total)\\% Processor Time').CounterSamples.CookedValue)",
        ]
    )
    return _parse_float(output)


def _nvidia_runtime_metrics() -> dict[str, dict[str, float]]:
    output = _run_command(
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
        gpu_value = _parse_float(gpu_usage)
        used_value = _parse_float(memory_used)
        total_value = _parse_float(memory_total)
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


def _site_package_roots() -> list[Path]:
    roots: list[Path] = []
    try:
        roots.append(Path(site.getusersitepackages()))
    except Exception:
        pass
    try:
        for entry in site.getsitepackages():
            roots.append(Path(entry))
    except Exception:
        pass

    unique: list[Path] = []
    for root in roots:
        if root not in unique and root.exists():
            unique.append(root)
    return unique


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _cuda_component_paths() -> tuple[Path | None, Path | None, Path | None]:
    explicit_roots = [
        Path(value)
        for key in ("COLLATZ_CUDA_HOME", "CUDA_HOME", "CUDA_PATH")
        if (value := os.getenv(key))
    ]

    nvvm_candidates: list[Path] = []
    libdevice_candidates: list[Path] = []
    cudart_candidates: list[Path] = []

    for root in explicit_roots:
        nvvm_candidates.extend(
            [
                root / "nvvm" / "bin" / "nvvm.dll",
                root / "nvvm" / "bin" / "nvvm64_40_0.dll",
                root / "bin" / "x86_64" / "nvvm64_40_0.dll",
            ]
        )
        libdevice_candidates.append(root / "nvvm" / "libdevice" / "libdevice.10.bc")
        cudart_candidates.extend(sorted((root / "bin").glob("cudart64_*.dll")))

    for site_root in _site_package_roots():
        nvcc_root = site_root / "nvidia" / "cuda_nvcc"
        compatibility_root = site_root / "nvidia" / "cu13"
        runtime_root = site_root / "nvidia" / "cuda_runtime"

        nvvm_candidates.extend(
            [
                nvcc_root / "nvvm" / "bin" / "nvvm64_40_0.dll",
                compatibility_root / "bin" / "x86_64" / "nvvm64_40_0.dll",
            ]
        )
        libdevice_candidates.extend(
            [
                nvcc_root / "nvvm" / "libdevice" / "libdevice.10.bc",
                compatibility_root / "nvvm" / "libdevice" / "libdevice.10.bc",
            ]
        )
        cudart_candidates.extend(sorted((runtime_root / "bin").glob("cudart64_*.dll")))

    return (
        _first_existing(nvvm_candidates),
        _first_existing(libdevice_candidates),
        _first_existing(cudart_candidates),
    )


def _copy_if_needed(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        same_size = target.stat().st_size == source.stat().st_size
        same_mtime = int(target.stat().st_mtime) == int(source.stat().st_mtime)
        if same_size and same_mtime:
            return
    shutil.copy2(source, target)


def _register_dll_directory(path: Path) -> None:
    if hasattr(os, "add_dll_directory") and path.exists():
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(path)))


def _prepare_cuda_home() -> Path | None:
    nvvm_source, libdevice_source, cudart_source = _cuda_component_paths()
    if not nvvm_source or not libdevice_source or not cudart_source:
        return None

    shim_root = Path(tempfile.gettempdir()) / "collatz-lab-cuda-shim"
    _copy_if_needed(nvvm_source, shim_root / "nvvm" / "bin" / "nvvm.dll")
    _copy_if_needed(libdevice_source, shim_root / "nvvm" / "libdevice" / "libdevice.10.bc")
    _copy_if_needed(cudart_source, shim_root / "bin" / cudart_source.name)

    os.environ["CUDA_HOME"] = str(shim_root)
    os.environ["CUDA_PATH"] = str(shim_root)
    _register_dll_directory(shim_root / "nvvm" / "bin")
    _register_dll_directory(shim_root / "bin")
    return shim_root


_prepare_cuda_home()

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


def gpu_execution_ready() -> bool:
    return bool(gpu_execution_diagnostics()["ready"])


def _detect_cpu_label() -> str:
    label = platform.processor().strip()
    if label:
        return label
    for key in ("PROCESSOR_IDENTIFIER", "PROCESSOR_ARCHITECTURE"):
        value = os.getenv(key, "").strip()
        if value:
            return value
    return "CPU"


def _detect_nvidia_gpus() -> list[HardwareCapability]:
    output = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return []

    diagnostics = gpu_execution_diagnostics()
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
                supported_kernels=[GPU_KERNEL] if diagnostics["ready"] else [],
                metadata={
                    "vendor": "nvidia",
                    "index": index,
                    "memory_mib": int(memory_mib),
                    "driver_version": driver_version,
                    "execution_ready": diagnostics["ready"],
                    "diagnostic": diagnostics["reason"],
                    "cuda_home": diagnostics["cuda_home"],
                    **live_metrics,
                },
            )
        )
    return capabilities


def discover_hardware() -> list[HardwareCapability]:
    cpu_usage = _cpu_usage_percent()
    capabilities = [
        HardwareCapability(
            kind="cpu",
            slug="cpu-default",
            label=_detect_cpu_label(),
            available=True,
            supported_hardware=["cpu"],
            supported_kernels=CPU_KERNELS,
            metadata={
                "logical_cores": os.cpu_count() or 1,
                "execution_ready": True,
                **({"usage_percent": cpu_usage} if cpu_usage is not None else {}),
            },
        )
    ]
    capabilities.extend(_detect_nvidia_gpus())
    return capabilities


def select_worker_execution_profile(
    capabilities: list[HardwareCapability],
    requested_hardware: str,
) -> tuple[list[str], list[str]]:
    hardware_targets: set[str] = set()
    kernel_targets: set[str] = set()

    for capability in capabilities:
        if not capability.available or not capability.supported_kernels:
            continue
        if requested_hardware != "auto" and requested_hardware not in capability.supported_hardware:
            continue
        hardware_targets.update(capability.supported_hardware)
        kernel_targets.update(capability.supported_kernels)

    if not hardware_targets or not kernel_targets:
        raise ValueError(
            f"No executable worker profile is available for hardware='{requested_hardware}'."
        )
    return sorted(hardware_targets), sorted(kernel_targets)


def validate_execution_request(
    *,
    requested_hardware: str,
    requested_kernel: str,
    capabilities: list[HardwareCapability] | None = None,
) -> None:
    inventory = capabilities or discover_hardware()
    supported_hardware, supported_kernels = select_worker_execution_profile(
        inventory,
        requested_hardware=requested_hardware,
    )
    if requested_hardware != "auto" and requested_hardware not in supported_hardware:
        raise ValueError(
            f"Hardware '{requested_hardware}' is not executable on this machine right now."
        )
    if requested_kernel not in supported_kernels:
        raise ValueError(
            f"Kernel '{requested_kernel}' cannot run on hardware '{requested_hardware}' yet."
        )
