"""Host metrics (CPU usage, etc.) — best-effort per OS; missing data is OK."""

from __future__ import annotations

import platform
import time

from .util import parse_float, run_command


def _cpu_usage_psutil() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    try:
        return float(psutil.cpu_percent(interval=0.12))
    except Exception:
        return None


def _cpu_usage_windows_perf() -> float | None:
    output = run_command(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "[string]::Format([System.Globalization.CultureInfo]::InvariantCulture, '{0:F2}', (Get-Counter '\\Processor(_Total)\\% Processor Time').CounterSamples.CookedValue)",
        ]
    )
    return parse_float(output)


def _read_linux_cpu_jiffies() -> tuple[int, int] | None:
    try:
        with open("/proc/stat", encoding="utf-8") as handle:
            line = handle.readline()
    except OSError:
        return None
    parts = line.split()
    if len(parts) < 8 or parts[0] != "cpu":
        return None
    try:
        nums = [int(p) for p in parts[1:8]]
    except ValueError:
        return None
    idle = nums[3] + nums[4]
    total = sum(nums)
    return idle, total


def _cpu_usage_linux_proc_stat() -> float | None:
    if platform.system() != "Linux":
        return None
    first = _read_linux_cpu_jiffies()
    if first is None:
        return None
    time.sleep(0.08)
    second = _read_linux_cpu_jiffies()
    if second is None:
        return None
    idle_delta = second[0] - first[0]
    total_delta = second[1] - first[1]
    if total_delta <= 0:
        return None
    busy = 100.0 * (1.0 - (idle_delta / total_delta))
    return round(max(0.0, min(100.0, busy)), 2)


def cpu_usage_percent_and_source() -> tuple[float | None, str | None]:
    """
    Aggregate CPU usage plus which probe succeeded (for smart_detection metadata).
    """
    value = _cpu_usage_psutil()
    if value is not None:
        return value, "psutil"
    system = platform.system()
    if system == "Windows":
        w = _cpu_usage_windows_perf()
        if w is not None:
            return w, "windows_perf_counter"
        return None, None
    if system == "Linux":
        l = _cpu_usage_linux_proc_stat()
        if l is not None:
            return l, "linux_proc_stat_delta"
        return None, None
    return None, None


def cpu_usage_percent() -> float | None:
    """Aggregate CPU usage (all cores). Order: psutil (optional), OS-specific probes."""
    return cpu_usage_percent_and_source()[0]
