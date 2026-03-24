"""
Windows CUDA DLL discovery and temp shim (Numba NVVM / cudart).
Must run before importing numba.cuda — see gpu.ensure_cuda_shim().
"""

from __future__ import annotations

import os
import shutil
import site
import tempfile
from collections.abc import Iterable
from pathlib import Path

_DLL_DIRECTORY_HANDLES: list[object] = []
_SHIM_INITIALIZED = False


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


def ensure_cuda_shim() -> Path | None:
    """Idempotent: run once before numba.cuda import on Windows wheel stacks."""
    global _SHIM_INITIALIZED
    if _SHIM_INITIALIZED:
        home = os.getenv("CUDA_HOME")
        return Path(home) if home else None
    _SHIM_INITIALIZED = True
    return _prepare_cuda_home()
