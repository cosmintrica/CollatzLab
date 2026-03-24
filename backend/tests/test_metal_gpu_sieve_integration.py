"""End-to-end: native Metal ``gpu-sieve`` streaming matches PyTorch MPS (macOS + helper binary)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="Metal gpu-sieve is macOS-only")


def test_metal_gpu_sieve_streaming_matches_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[2]
    metal_dir = root / "scripts" / "native_sieve_kit" / "metal"
    try:
        subprocess.run(
            ["bash", str(metal_dir / "build_metal_sieve_chunk.sh")],
            cwd=str(metal_dir),
            check=True,
            capture_output=True,
            timeout=120,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("Metal chunk build failed or no bash")

    bin_path = metal_dir / "metal_sieve_chunk"
    if not bin_path.is_file():
        pytest.skip("metal_sieve_chunk missing")

    monkeypatch.setenv("COLLATZ_LAB_ROOT", str(root))
    from collatz_lab.gpu_sieve_metal_runtime import metal_sieve_chunk_binary_path

    metal_sieve_chunk_binary_path.cache_clear()
    if metal_sieve_chunk_binary_path() is None:
        pytest.skip("metal_sieve_chunk not executable on PATH resolution")

    try:
        import torch

        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")
    except Exception:
        pytest.skip("torch/mps not available")

    from collatz_lab.services import compute_range_metrics_gpu_sieve

    monkeypatch.setenv("COLLATZ_GPU_SIEVE_BACKEND", "mps")
    metal_sieve_chunk_binary_path.cache_clear()
    mps_agg = compute_range_metrics_gpu_sieve(1, 19_999)

    monkeypatch.setenv("COLLATZ_GPU_SIEVE_BACKEND", "metal")
    metal_sieve_chunk_binary_path.cache_clear()
    metal_agg = compute_range_metrics_gpu_sieve(1, 19_999)

    assert mps_agg.processed == metal_agg.processed
    assert mps_agg.last_processed == metal_agg.last_processed
    assert mps_agg.max_total_stopping_time == metal_agg.max_total_stopping_time
    assert mps_agg.max_stopping_time == metal_agg.max_stopping_time
    assert mps_agg.max_excursion == metal_agg.max_excursion
