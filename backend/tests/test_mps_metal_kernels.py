"""Strict parity tests for Apple MPS kernels (Metal via PyTorch).

Skipped automatically on non-macOS or when MPS is unavailable (e.g. CI Linux).
"""
from __future__ import annotations

import pytest

from collatz_lab.hardware import GPU_KERNEL, GPU_SIEVE_KERNEL

try:
    from collatz_lab import mps_collatz
except Exception:  # pragma: no cover
    mps_collatz = None

from collatz_lab.services import compute_range_metrics, metrics_direct


def _mps_runtime_ok() -> bool:
    return bool(mps_collatz and mps_collatz.mps_accelerated_available())


pytestmark = pytest.mark.skipif(
    not _mps_runtime_ok(),
    reason="MPS (Apple Metal) not available",
)


class TestMpsAcceleratedVsCpuDirect:
    def test_ranges_match_cpu_direct(self):
        for start, end in ((1, 5000), (10_000, 11_000), (99_001, 100_000)):
            g = compute_range_metrics(start, end, kernel=GPU_KERNEL)
            d = compute_range_metrics(start, end, kernel="cpu-direct")
            assert g.max_total_stopping_time == d.max_total_stopping_time, f"[{start},{end}] max_total"
            assert g.max_stopping_time == d.max_stopping_time, f"[{start},{end}] max_stopping"
            assert g.max_excursion == d.max_excursion, f"[{start},{end}] max_excursion"
            assert g.processed == d.processed

    def test_single_seeds(self):
        for n in (1, 2, 27, 9663, 100_001):
            g = compute_range_metrics(n, n, kernel=GPU_KERNEL)
            ref = metrics_direct(n)
            assert g.max_total_stopping_time["value"] == ref.total_stopping_time
            assert g.max_stopping_time["value"] == ref.stopping_time
            assert g.max_excursion["value"] == ref.max_excursion


class TestMpsSieveVsCpuSieve:
    def test_ranges_match_cpu_sieve(self):
        for start, end in ((1, 3000), (1, 8000), (50_001, 52_000)):
            gs = compute_range_metrics(start, end, kernel=GPU_SIEVE_KERNEL)
            cs = compute_range_metrics(start, end, kernel="cpu-sieve")
            assert gs.max_total_stopping_time == cs.max_total_stopping_time, f"[{start},{end}] max_total"
            assert gs.max_stopping_time == cs.max_stopping_time
            assert gs.max_excursion == cs.max_excursion
            assert gs.processed == cs.processed

    def test_gpu_sieve_is_not_cpu_only_path(self):
        """gpu-sieve must use MPS tensors when MPS is available (regression guard)."""
        import torch

        a = torch.arange(3, 15, 2, dtype=torch.int64, device=torch.device("mps"))
        t, s, m = mps_collatz._mps_batch_sieve_descent(a)
        assert t.device.type == "mps"
        assert int(t[0].item()) >= 0
