from __future__ import annotations

import pytest

from collatz_lab.services import (
    COLLATZ_INT64_ODD_STEP_LIMIT,
    accelerated_odd_step,
    collatz_step,
    compute_range_metrics,
    metrics_accelerated,
    metrics_direct,
)


def test_collatz_step_basic_cases():
    assert collatz_step(1) == 4
    assert collatz_step(2) == 1
    assert collatz_step(3) == 10


def test_accelerated_odd_step_counts_divisions():
    next_value, shifts = accelerated_odd_step(3)
    assert next_value == 5
    assert shifts == 1


def test_direct_and_accelerated_metrics_match_small_interval():
    for value in range(1, 250):
        assert metrics_direct(value) == metrics_accelerated(value)


def test_cpu_parallel_kernel_matches_direct_interval_aggregate():
    direct = compute_range_metrics(1, 250, kernel="cpu-direct")
    parallel = compute_range_metrics(1, 250, kernel="cpu-parallel")

    assert parallel.processed == direct.processed
    assert parallel.last_processed == direct.last_processed
    assert parallel.max_total_stopping_time == direct.max_total_stopping_time
    assert parallel.max_stopping_time == direct.max_stopping_time
    assert parallel.max_excursion == direct.max_excursion


def test_cpu_parallel_recovers_overflow_via_bigint_fallback():
    overflow_seed = COLLATZ_INT64_ODD_STEP_LIMIT + 1
    result = compute_range_metrics(overflow_seed, overflow_seed, kernel="cpu-parallel")
    direct = metrics_accelerated(overflow_seed)
    assert result.processed == 1
    assert result.max_total_stopping_time["value"] == direct.total_stopping_time
    assert result.max_stopping_time["value"] == direct.stopping_time
