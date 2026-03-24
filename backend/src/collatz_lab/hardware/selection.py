"""Map requested hardware / kernel to executable worker profile."""

from __future__ import annotations

from ..schemas import HardwareCapability
from .discovery import discover_hardware


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
