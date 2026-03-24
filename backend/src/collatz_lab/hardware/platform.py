"""Normalized host OS / ISA context for capability metadata (Phase A portability)."""

from __future__ import annotations

import platform
from typing import Any


def platform_context() -> dict[str, Any]:
    """
    Serializable host facts for reproducibility and future kernel routing.
    Does not change execution paths — metadata only.
    """
    system = platform.system().lower()
    machine = (platform.machine() or "").lower()
    uname = platform.uname()
    return {
        "os": system,
        "machine": machine,
        "processor": (platform.processor() or "").strip(),
        "python_version": platform.python_version(),
        "node": uname.node,
        "is_windows": system == "windows",
        "is_linux": system == "linux",
        "is_darwin": system == "darwin",
        "is_x86_64": machine in ("x86_64", "amd64"),
        "is_arm64": machine in ("arm64", "aarch64"),
    }
