"""Best-effort CPU marketing / model string per OS (baseline Windows behavior unchanged)."""

from __future__ import annotations

import os
import platform

from .util import run_command


def detect_cpu_label() -> str:
    sys = platform.system().lower()

    if sys == "darwin":
        # Prefer marketing name; sysctl path is stable (works from GUI apps with reduced PATH).
        out = run_command(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"])
        if not out:
            out = run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out.strip()

    if sys == "linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    line_lower = line.lower()
                    if line_lower.startswith("model name") or line_lower.startswith("model\t"):
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            candidate = parts[1].strip()
                            if candidate:
                                return candidate
        except OSError:
            pass

    label = platform.processor().strip()
    if label:
        return label
    for key in ("PROCESSOR_IDENTIFIER", "PROCESSOR_ARCHITECTURE"):
        value = os.getenv(key, "").strip()
        if value:
            return value
    return "CPU"
