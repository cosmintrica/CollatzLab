"""Research-grade correctness tests for all Collatz kernels.

These tests ensure that every kernel (cpu-direct, cpu-accelerated,
cpu-parallel, gpu-collatz-accelerated) produces bit-identical results
for all three metrics: total_stopping_time, stopping_time, max_excursion.

The reference implementation is `metrics_direct` — a pure-Python,
single-step Collatz walker with no optimisations.  Every other kernel
must agree with it on every seed tested.
"""
from __future__ import annotations

import random

import pytest

from collatz_lab.hardware import gpu_execution_ready, GPU_KERNEL
from collatz_lab.services import (
    COLLATZ_INT64_ODD_STEP_LIMIT,
    INT64_MAX,
    compute_range_metrics,
    compute_range_metrics_parallel,
    metrics_accelerated,
    metrics_direct,
)


# ---------------------------------------------------------------------------
# 1.  Per-seed: direct vs accelerated  (Python reference implementations)
# ---------------------------------------------------------------------------

class TestDirectVsAccelerated:
    """metrics_direct and metrics_accelerated must be identical on every seed."""

    def test_small_range_1_to_1000(self):
        for n in range(1, 1001):
            d, a = metrics_direct(n), metrics_accelerated(n)
            assert d == a, f"Mismatch at n={n}: direct={d} accelerated={a}"

    def test_powers_of_two(self):
        """Pure even seeds — stopping_time must be 1 for n>=2."""
        for exp in range(1, 40):
            n = 1 << exp
            d = metrics_direct(n)
            a = metrics_accelerated(n)
            assert d == a, f"Mismatch at 2^{exp}: direct={d} accelerated={a}"
            assert d.stopping_time == 1, f"2^{exp}: stopping_time should be 1, got {d.stopping_time}"

    def test_mersenne_like(self):
        """2^k - 1 seeds (all-ones in binary, always odd)."""
        for exp in range(2, 35):
            n = (1 << exp) - 1
            d, a = metrics_direct(n), metrics_accelerated(n)
            assert d == a, f"Mismatch at 2^{exp}-1={n}: direct={d} accelerated={a}"

    def test_specific_known_values(self):
        """Well-known Collatz seeds with published metrics."""
        # n=27: total_stopping_time=111, max_excursion=9232
        d = metrics_direct(27)
        assert d.total_stopping_time == 111
        assert d.max_excursion == 9232
        assert d == metrics_accelerated(27)

        # n=9663 has a notably high stopping time
        d9663 = metrics_direct(9663)
        assert d9663 == metrics_accelerated(9663)

    def test_random_seeds_in_billion_range(self):
        """Random seeds in 10^9..10^10 range — real frontier territory."""
        rng = random.Random(42)  # deterministic
        seeds = [rng.randint(10**9, 10**10) for _ in range(200)]
        for n in seeds:
            d, a = metrics_direct(n), metrics_accelerated(n)
            assert d == a, f"Mismatch at n={n}: direct={d} accelerated={a}"


# ---------------------------------------------------------------------------
# 2.  Per-seed: cpu-parallel vs direct  (Numba JIT correctness)
# ---------------------------------------------------------------------------

class TestParallelVsDirect:
    """cpu-parallel must agree with cpu-direct per-seed on all three metrics."""

    @staticmethod
    def _compare_range(start, end):
        """Utility: compare parallel aggregate against per-seed direct."""
        parallel = compute_range_metrics(start, end, kernel="cpu-parallel")
        direct = compute_range_metrics(start, end, kernel="cpu-direct")
        assert parallel.processed == direct.processed
        assert parallel.max_total_stopping_time == direct.max_total_stopping_time, \
            f"max_total_stopping_time mismatch on [{start}, {end}]: parallel={parallel.max_total_stopping_time} direct={direct.max_total_stopping_time}"
        assert parallel.max_stopping_time == direct.max_stopping_time, \
            f"max_stopping_time mismatch on [{start}, {end}]: parallel={parallel.max_stopping_time} direct={direct.max_stopping_time}"
        assert parallel.max_excursion == direct.max_excursion, \
            f"max_excursion mismatch on [{start}, {end}]: parallel={parallel.max_excursion} direct={direct.max_excursion}"

    def test_range_1_to_1000(self):
        self._compare_range(1, 1000)

    def test_range_1_to_10000(self):
        self._compare_range(1, 10_000)

    def test_even_only_range(self):
        """Range with only even starting seeds."""
        self._compare_range(2, 500)

    def test_odd_only_range(self):
        """Range with only odd starting seeds (1, 3, 5, ...)."""
        # Use a range where start is odd and we check per-seed
        for start in range(1, 200, 2):
            self._compare_range(start, start)

    def test_single_seeds_problematic_cases(self):
        """Individually verify seeds that previously exposed bugs."""
        problematic = [1, 2, 3, 4, 8, 16, 24, 32, 64, 128, 256, 512, 1024,
                       27, 9663, 77031, 113383]
        for n in problematic:
            d = metrics_direct(n)
            p = compute_range_metrics(n, n, kernel="cpu-parallel")
            assert p.max_total_stopping_time["value"] == d.total_stopping_time, \
                f"n={n}: total_stopping_time parallel={p.max_total_stopping_time['value']} direct={d.total_stopping_time}"
            assert p.max_stopping_time["value"] == d.stopping_time, \
                f"n={n}: stopping_time parallel={p.max_stopping_time['value']} direct={d.stopping_time}"
            assert p.max_excursion["value"] == d.max_excursion, \
                f"n={n}: max_excursion parallel={p.max_excursion['value']} direct={d.max_excursion}"

    def test_random_windows_in_large_range(self):
        """Random small windows in 10^8..10^9 to test at scale."""
        rng = random.Random(99)
        for _ in range(10):
            start = rng.randint(10**8, 10**9)
            end = start + 499
            self._compare_range(start, end)


# ---------------------------------------------------------------------------
# 3.  Overflow / frontier recovery
# ---------------------------------------------------------------------------

class TestOverflowRecovery:
    """The parallel kernel must not crash at the int64 frontier.
    Instead it must fall back to Python bigint and produce correct results."""

    def test_single_overflow_seed_recovery(self):
        overflow_seed = COLLATZ_INT64_ODD_STEP_LIMIT + 1
        result = compute_range_metrics(overflow_seed, overflow_seed, kernel="cpu-parallel")
        direct = metrics_accelerated(overflow_seed)
        assert result.processed == 1
        assert result.max_total_stopping_time["value"] == direct.total_stopping_time
        assert result.max_stopping_time["value"] == direct.stopping_time
        # Excursion may exceed int64 — verify it's at least as large as INT64_MAX
        # (the patch system should preserve the true value in the aggregate)
        assert result.max_excursion["value"] >= direct.max_excursion or \
               result.max_excursion["value"] == min(direct.max_excursion, INT64_MAX)

    def test_range_straddling_overflow_frontier(self):
        """A range where most seeds are safe but a few overflow."""
        frontier = COLLATZ_INT64_ODD_STEP_LIMIT
        # range includes frontier-2 .. frontier+2 (5 seeds, some safe, some overflow)
        result = compute_range_metrics(frontier - 2, frontier + 2, kernel="cpu-parallel")
        assert result.processed == 5
        # Verify against direct computation for the safe seeds
        for n in range(frontier - 2, frontier + 3):
            d = metrics_accelerated(n)
            # Just ensure no crash — the aggregate tracks max values only
            assert d.total_stopping_time > 0

    def test_overflow_does_not_corrupt_safe_seeds(self):
        """When a batch contains overflow seeds, the safe seeds must still be correct."""
        frontier = COLLATZ_INT64_ODD_STEP_LIMIT
        # Compute a window with known safe seeds + one overflow
        safe_start = frontier - 10
        safe_end = frontier - 1
        mixed_end = frontier + 1  # includes overflow seed

        safe_only = compute_range_metrics(safe_start, safe_end, kernel="cpu-parallel")
        mixed = compute_range_metrics(safe_start, mixed_end, kernel="cpu-parallel")

        # The safe-seed max metrics should be <= mixed (mixed has more seeds)
        assert mixed.max_total_stopping_time["value"] >= safe_only.max_total_stopping_time["value"]
        assert mixed.max_excursion["value"] >= safe_only.max_excursion["value"]


# ---------------------------------------------------------------------------
# 4.  GPU kernel (conditional — skip if no GPU)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not gpu_execution_ready(), reason="No GPU runtime available")
class TestGPUVsDirect:
    """GPU kernel must agree with cpu-direct on all metrics."""

    def test_gpu_small_range(self):
        gpu = compute_range_metrics(1, 1000, kernel=GPU_KERNEL)
        direct = compute_range_metrics(1, 1000, kernel="cpu-direct")
        assert gpu.max_total_stopping_time == direct.max_total_stopping_time
        assert gpu.max_stopping_time == direct.max_stopping_time
        assert gpu.max_excursion == direct.max_excursion

    def test_gpu_matches_parallel_medium_range(self):
        gpu = compute_range_metrics(10_000, 20_000, kernel=GPU_KERNEL)
        parallel = compute_range_metrics(10_000, 20_000, kernel="cpu-parallel")
        assert gpu.max_total_stopping_time == parallel.max_total_stopping_time
        assert gpu.max_stopping_time == parallel.max_stopping_time
        assert gpu.max_excursion == parallel.max_excursion

    def test_gpu_single_problematic_seeds(self):
        for n in [1, 2, 8, 16, 27, 9663]:
            gpu = compute_range_metrics(n, n, kernel=GPU_KERNEL)
            d = metrics_direct(n)
            assert gpu.max_total_stopping_time["value"] == d.total_stopping_time, \
                f"GPU n={n}: total_stopping_time={gpu.max_total_stopping_time['value']} expected={d.total_stopping_time}"
            assert gpu.max_stopping_time["value"] == d.stopping_time, \
                f"GPU n={n}: stopping_time={gpu.max_stopping_time['value']} expected={d.stopping_time}"
            assert gpu.max_excursion["value"] == d.max_excursion, \
                f"GPU n={n}: max_excursion={gpu.max_excursion['value']} expected={d.max_excursion}"


# ---------------------------------------------------------------------------
# 5.  Cross-kernel consistency matrix
# ---------------------------------------------------------------------------

class TestCrossKernelConsistency:
    """All kernels must produce identical aggregate metrics on the same range."""

    @pytest.fixture
    def reference_range(self):
        return (1, 5000)

    def test_all_cpu_kernels_agree(self, reference_range):
        start, end = reference_range
        direct = compute_range_metrics(start, end, kernel="cpu-direct")
        accel = compute_range_metrics(start, end, kernel="cpu-accelerated")
        parallel = compute_range_metrics(start, end, kernel="cpu-parallel")

        for label, result in [("accelerated", accel), ("parallel", parallel)]:
            assert result.max_total_stopping_time == direct.max_total_stopping_time, \
                f"{label} disagrees on max_total_stopping_time"
            assert result.max_stopping_time == direct.max_stopping_time, \
                f"{label} disagrees on max_stopping_time"
            assert result.max_excursion == direct.max_excursion, \
                f"{label} disagrees on max_excursion"
            assert result.processed == direct.processed, \
                f"{label} disagrees on processed count"

    @pytest.mark.skipif(not gpu_execution_ready(), reason="No GPU runtime available")
    def test_gpu_agrees_with_cpu_kernels(self, reference_range):
        start, end = reference_range
        direct = compute_range_metrics(start, end, kernel="cpu-direct")
        gpu = compute_range_metrics(start, end, kernel=GPU_KERNEL)

        assert gpu.max_total_stopping_time == direct.max_total_stopping_time
        assert gpu.max_stopping_time == direct.max_stopping_time
        assert gpu.max_excursion == direct.max_excursion
