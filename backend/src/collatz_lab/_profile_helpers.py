"""Compute-profile utility helpers shared by services, scheduling, and orchestration.

Intentionally self-contained — only imports from the standard library and
``collatz_lab.schemas`` (which has no package-level circular dependencies).
"""
from __future__ import annotations

import os

from .schemas import ComputeProfile


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _effective_profile_percent(profile: ComputeProfile | None, hardware: str) -> int:
    system_percent = 100 if profile is None else max(0, min(100, int(profile.system_percent)))
    hardware_percent = 100
    if profile is not None:
        hardware_percent = max(
            0,
            min(100, int(profile.cpu_percent if hardware == "cpu" else profile.gpu_percent)),
        )
    return max(0, min(100, round((system_percent * hardware_percent) / 100)))
