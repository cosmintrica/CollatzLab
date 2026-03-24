"""Parity: pure-Python ``sieve_reference`` vs Numba ``cpu-sieve`` (no overflow patch in ref)."""

from __future__ import annotations

import pytest

from collatz_lab.services import compute_range_metrics_sieve_odd
from collatz_lab.sieve_reference import odd_sieve_descent_linear_range, odd_sieve_descent_one


@pytest.mark.parametrize("end", [99, 500, 2000, 9999])
def test_reference_matches_cpu_sieve_on_small_linear_ranges(end: int):
    ref = odd_sieve_descent_linear_range(1, end)
    numba = compute_range_metrics_sieve_odd(1, end)
    assert ref["processed"] == numba.processed
    assert ref["last_processed"] == numba.last_processed
    assert ref["max_total_stopping_time"] == numba.max_total_stopping_time
    assert ref["max_stopping_time"] == numba.max_stopping_time
    assert ref["max_excursion"] == numba.max_excursion


def test_reference_single_seeds_sample():
    for seed in [1, 3, 27, 9663, 999_983]:
        a, b, c = odd_sieve_descent_one(seed)
        m = compute_range_metrics_sieve_odd(seed, seed)
        assert m.processed == 1
        assert m.max_total_stopping_time["n"] == seed
        assert m.max_total_stopping_time["value"] == a
        assert m.max_stopping_time["value"] == b
        assert m.max_excursion["value"] == c
