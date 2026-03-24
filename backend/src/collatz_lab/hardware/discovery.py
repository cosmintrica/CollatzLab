"""Aggregate CPU + GPU hardware capabilities for API and workers."""

from __future__ import annotations

import os
from typing import Any

from ..schemas import HardwareCapability
from .constants import CPU_KERNELS
from .cpu_label import detect_cpu_label
from .gpu_inventory import collect_gpu_capabilities
from .metrics import cpu_usage_percent_and_source
from .platform import platform_context
from .smart_detection import cpu_smart_detection


def discover_hardware() -> list[HardwareCapability]:
    usage, usage_source = cpu_usage_percent_and_source()
    host = platform_context()
    cpu_meta = {
        "logical_cores": os.cpu_count() or 1,
        "execution_ready": True,
        "host": host,
        **({"usage_percent": usage} if usage is not None else {}),
        **cpu_smart_detection(
            probes=[
                "detect_cpu_label",
                "os.cpu_count",
                "platform_context",
                "cpu_usage_probe_chain",
            ],
            usage_probe=usage_source,
            extra_signals={
                "cpu_label_source": _cpu_label_source_hint(host),
            },
        ),
    }
    capabilities = [
        HardwareCapability(
            kind="cpu",
            slug="cpu-default",
            label=detect_cpu_label(),
            available=True,
            supported_hardware=["cpu"],
            supported_kernels=CPU_KERNELS,
            metadata=cpu_meta,
        )
    ]
    capabilities.extend(collect_gpu_capabilities())
    return capabilities


def _cpu_label_source_hint(host: dict[str, Any]) -> str:
    """Lightweight hint for dev JSON (which branch likely ran inside detect_cpu_label)."""
    system = (host.get("os") or "").lower()
    if system == "darwin":
        return "sysctl_machdep_cpu_brand_or_platform_processor"
    if system == "linux":
        return "proc_cpuinfo_or_platform_processor"
    return "platform_processor_or_env"

