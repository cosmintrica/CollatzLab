"""Cross-platform hardware discovery and metrics (no CUDA required)."""

from __future__ import annotations

from collatz_lab.hardware.discovery import discover_hardware
from collatz_lab.hardware.gpu_inventory import collect_gpu_capabilities
from collatz_lab.schemas import HardwareCapability


def test_discover_hardware_merges_gpu_inventory(monkeypatch):
    def fake_collect() -> list[HardwareCapability]:
        return [
            HardwareCapability(
                kind="gpu",
                slug="gpu-simulated-igpu",
                label="Simulated integrated GPU",
                available=True,
                supported_hardware=[],
                supported_kernels=[],
                metadata={"diagnostic": "test-diagnostic"},
            )
        ]

    monkeypatch.setattr("collatz_lab.hardware.discovery.collect_gpu_capabilities", fake_collect)
    caps = discover_hardware()
    assert any(c.slug == "gpu-simulated-igpu" for c in caps)
    gpu = next(c for c in caps if c.slug == "gpu-simulated-igpu")
    assert gpu.supported_kernels == []


def test_discover_cpu_includes_smart_detection():
    caps = discover_hardware()
    cpu = next(c for c in caps if c.kind == "cpu")
    sd = cpu.metadata.get("smart_detection")
    assert sd is not None
    assert sd.get("schema_version") == 1
    assert "collatz_cpu_executable" in sd
    assert sd["collatz_cpu_executable"] is True


def test_gpu_inventory_dedupes_nvidia_display_rows(monkeypatch):
    nvidia_rows = [
        HardwareCapability(
            kind="gpu",
            slug="gpu-nvidia-0",
            label="NVIDIA GeForce RTX 5090",
            available=True,
            supported_hardware=["gpu"],
            supported_kernels=["gpu-sieve"],
            metadata={"vendor": "nvidia"},
        )
    ]
    display_rows = [
        HardwareCapability(
            kind="gpu",
            slug="gpu-win-0",
            label="Intel UHD 770",
            available=True,
            supported_hardware=[],
            supported_kernels=[],
            metadata={"vendor": "intel"},
        ),
        HardwareCapability(
            kind="gpu",
            slug="gpu-win-1",
            label="NVIDIA GeForce RTX 5090",
            available=True,
            supported_hardware=[],
            supported_kernels=[],
            metadata={"vendor": "nvidia"},
        ),
    ]

    monkeypatch.setattr("collatz_lab.hardware.gpu_inventory.detect_nvidia_gpus", lambda: nvidia_rows)
    monkeypatch.setattr("collatz_lab.hardware.gpu_inventory.probe_display_adapters", lambda: display_rows)

    merged = collect_gpu_capabilities()
    slugs = [c.slug for c in merged]
    assert "gpu-nvidia-0" in slugs
    assert "gpu-win-0" in slugs
    assert "gpu-win-1" not in slugs
    assert len([c for c in merged if c.kind == "gpu"]) == 2
