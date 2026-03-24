"""
Display / PCI GPU probes — always run as part of smart inventory.

Rows are merged with nvidia-smi results in ``gpu_inventory`` (dedupe).
"""

from __future__ import annotations

import platform
import re
import shutil
from functools import lru_cache

from ...schemas import HardwareCapability
from ..smart_detection import display_probe_smart_detection
from ..util import run_command

_COLLATZ_CUDA_ONLY = (
    "Collatz GPU kernels are CUDA/Numba-only in this release. "
    "This device is visible to the OS but has no executable lab kernel here yet."
)


def _display_gpu_capability(
    *,
    slug: str,
    label: str,
    vendor: str,
    probe_tool: str,
    raw_snippet: str | None = None,
) -> HardwareCapability:
    smart = display_probe_smart_detection(
        vendor=vendor,
        probe_tool=probe_tool,
        collatz_gpu_executable=False,
        raw_snippet=raw_snippet,
    )
    return HardwareCapability(
        kind="gpu",
        slug=slug,
        label=label,
        available=True,
        supported_hardware=[],
        supported_kernels=[],
        metadata={
            "vendor": vendor,
            "execution_ready": False,
            "collatz_gpu_backend": "none",
            "diagnostic": _COLLATZ_CUDA_ONLY,
            **smart,
        },
    )


def _parse_system_profiler_displays(text: str) -> list[str]:
    labels: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Chipset Model:"):
            name = stripped.split(":", 1)[1].strip()
            if name and name not in labels:
                labels.append(name)
    return labels


def _parse_ioreg_gpu_models(text: str) -> list[str]:
    """IOReg uses lines like:  "model" = <"Apple M3">"""
    found = re.findall(r'"model"\s*=\s*<"([^"]+)"\s*>', text)
    labels: list[str] = []
    for name in found:
        low = name.lower()
        if "display" in low and "apple" not in low:
            continue
        if name not in labels:
            labels.append(name)
    return labels


def _macos_apple_silicon_gpu_fallback_label() -> str | None:
    """When profiler/ioreg fail (PATH, sandbox), still expose a sane GPU row on Apple Silicon."""
    if platform.machine().lower() not in ("arm64", "aarch64"):
        return None
    gpu_flag = run_command(["/usr/sbin/sysctl", "-n", "hw.optional.gpu"])
    if not gpu_flag:
        gpu_flag = run_command(["sysctl", "-n", "hw.optional.gpu"])
    if (gpu_flag or "").strip() != "1":
        return None
    chip = run_command(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"])
    if not chip:
        chip = run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
    chip = (chip or "Apple Silicon").strip()
    return f"Apple GPU (integrated, with {chip})"


@lru_cache(maxsize=1)
def _macos_display_entries() -> tuple[tuple[str, str], ...]:
    """
    Returns (label, probe_key) tuples. probe_key is used for smart_detection.primary_signal mapping.
    """
    if platform.system() != "Darwin":
        return ()

    # system_profiler lives in /usr/sbin — often missing from non-login PATH (IDE, launchd).
    for cmd in (
        ["/usr/sbin/system_profiler", "SPDisplaysDataType"],
        ["system_profiler", "SPDisplaysDataType"],
    ):
        output = run_command(cmd, timeout=90.0)
        if output:
            labels = _parse_system_profiler_displays(output)
            if labels:
                return tuple((name, "system_profiler_SPDisplaysDataType") for name in labels)

    for cmd in (
        ["/usr/sbin/ioreg", "-r", "-c", "IOGPU", "-d", "2", "-w", "0"],
        ["ioreg", "-r", "-c", "IOGPU", "-d", "2", "-w", "0"],
    ):
        output = run_command(cmd, timeout=30.0)
        if output:
            labels = _parse_ioreg_gpu_models(output)
            if labels:
                return tuple((name, "ioreg_IOGPU") for name in labels)

    fallback = _macos_apple_silicon_gpu_fallback_label()
    if fallback:
        return ((fallback, "sysctl_apple_silicon_heuristic"),)

    return ()


def _vendor_from_label(label: str) -> str:
    low = label.lower()
    if "apple" in low:
        return "apple"
    if "amd" in low or "radeon" in low:
        return "amd"
    if "intel" in low:
        return "intel"
    if "nvidia" in low:
        return "nvidia"
    return "unknown"


def _probe_macos_displays() -> list[HardwareCapability]:
    entries = _macos_display_entries()
    if not entries:
        return []
    caps: list[HardwareCapability] = []
    for index, (label, probe_tool) in enumerate(entries):
        vendor = _vendor_from_label(label)
        caps.append(
            _display_gpu_capability(
                slug=f"gpu-macos-display-{index}",
                label=label,
                vendor=vendor,
                probe_tool=probe_tool,
                raw_snippet=label,
            )
        )
    return caps


def _probe_linux_pci() -> list[HardwareCapability]:
    if platform.system() != "Linux":
        return []
    if not shutil.which("lspci"):
        return []
    output = run_command(["lspci"])
    if not output:
        return []
    pattern = re.compile(r"vga|3d|display", re.IGNORECASE)
    seen: set[str] = set()
    caps: list[HardwareCapability] = []
    for line in output.splitlines():
        if not pattern.search(line):
            continue
        label = line.split(":", 2)[-1].strip() if ":" in line else line.strip()
        if not label or label in seen:
            continue
        seen.add(label)
        low = label.lower()
        if "nvidia" in low:
            vendor = "nvidia"
        elif "amd" in low or "ati" in low:
            vendor = "amd"
        elif "intel" in low:
            vendor = "intel"
        else:
            vendor = "unknown"
        caps.append(
            _display_gpu_capability(
                slug=f"gpu-pci-{len(caps)}",
                label=label,
                vendor=vendor,
                probe_tool="lspci_vga_class",
                raw_snippet=line.strip()[:240],
            )
        )
    return caps[:6]


def _probe_windows_video() -> list[HardwareCapability]:
    if platform.system() != "Windows":
        return []
    output = run_command(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }",
        ]
    )
    if not output:
        return []
    skip_tokens = ("microsoft basic render driver", "remote display adapter")
    caps: list[HardwareCapability] = []
    seen: set[str] = set()
    for raw in output.splitlines():
        name = raw.strip()
        if not name or name.lower() in seen:
            continue
        low = name.lower()
        if any(tok in low for tok in skip_tokens):
            continue
        seen.add(low)
        if "nvidia" in low:
            vendor = "nvidia"
        elif "amd" in low or "radeon" in low:
            vendor = "amd"
        elif "intel" in low:
            vendor = "intel"
        else:
            vendor = "unknown"
        caps.append(
            _display_gpu_capability(
                slug=f"gpu-win-{len(caps)}",
                label=name,
                vendor=vendor,
                probe_tool="cim_win32_videocontroller",
                raw_snippet=name,
            )
        )
    return caps[:6]


def probe_display_adapters() -> list[HardwareCapability]:
    """Run the OS-appropriate display / PCI probe (safe to call on every inventory refresh)."""
    system = platform.system()
    if system == "Darwin":
        return _probe_macos_displays()
    if system == "Linux":
        return _probe_linux_pci()
    if system == "Windows":
        return _probe_windows_video()
    return []
