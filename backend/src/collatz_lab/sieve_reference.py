"""Canonical **odd-only descent** semantics (mirror of ``_collatz_sieve_parallel_odd``).

Used for:

- proving native (C / Metal) and other fast paths match the lab’s **cpu-sieve** / **gpu-sieve** contract on **all platforms**;
- property tests without calling Numba internals.

This is **intentionally slow** pure Python — do not use in production compute paths.

Platform-wide validation context: ``collatz_lab.validation_source`` and ``docs/CORRECTNESS_AND_VALIDATION.md``.
"""

from __future__ import annotations

# Must stay in lockstep with ``services.py`` for INT64 / limits / loop shape.
INT64_MAX = (1 << 63) - 1
COLLATZ_INT64_ODD_STEP_LIMIT = (INT64_MAX - 1) // 3
MAX_KERNEL_STEPS = 100_000


def odd_sieve_descent_one(seed: int) -> tuple[int, int, int]:
    """One odd seed: standard Collatz steps until first value **< seed** (or trivial / overflow).

    Returns ``(total_steps, stopping_steps, max_excursion)`` matching Numba
    ``_collatz_sieve_parallel_odd`` (``total == stopping`` for this kernel).

    Overflow / step-cap failure: ``(-1, -1, -1)``.
    """
    if seed <= 1:
        return (0, 0, int(seed))

    current = int(seed)
    steps = 0
    max_excursion = current

    while current >= seed and steps < MAX_KERNEL_STEPS:
        if current & 1 == 0:
            current >>= 1
            steps += 1
        else:
            if current > COLLATZ_INT64_ODD_STEP_LIMIT:
                return (-1, -1, -1)
            current = 3 * current + 1
            steps += 1
            if current > max_excursion:
                max_excursion = current
            while current & 1 == 0 and current >= seed:
                current >>= 1
                steps += 1

    if current >= seed and current != 1 and steps >= MAX_KERNEL_STEPS:
        return (-1, -1, -1)

    return (steps, steps, max_excursion)


def odd_sieve_descent_linear_range(start: int, end: int) -> dict[str, object]:
    """Aggregate over odd seeds in ``[start, end]`` (same odd set as ``compute_range_metrics_sieve_odd``).

    **Note:** Does not implement Numba’s int64 overflow **patch** (``metrics_descent_direct`` fallback).
    Only use on intervals where the parallel kernel never returns ``-1`` — e.g. modest ``end``.

    Keys match aggregate fields used for parity checks: ``processed``, ``last_processed``,
    ``max_total_stopping_time``, ``max_stopping_time``, ``max_excursion``.
    """
    if start < 1 or end < start:
        raise ValueError("Invalid linear interval.")
    first_odd = start if start & 1 else start + 1
    if first_odd > end:
        return {
            "processed": 0,
            "last_processed": end,
            "max_total_stopping_time": {"n": start, "value": 0},
            "max_stopping_time": {"n": start, "value": 0},
            "max_excursion": {"n": start, "value": 0},
        }

    odd_count = ((end - first_odd) // 2) + 1
    best_tst = {"n": first_odd, "value": -1}
    best_st = {"n": first_odd, "value": -1}
    best_exc = {"n": first_odd, "value": -1}

    for i in range(odd_count):
        seed = first_odd + 2 * i
        tst, st, mx = odd_sieve_descent_one(seed)
        if tst > best_tst["value"]:
            best_tst = {"n": seed, "value": tst}
        if st > best_st["value"]:
            best_st = {"n": seed, "value": st}
        if mx > best_exc["value"]:
            best_exc = {"n": seed, "value": mx}

    return {
        "processed": odd_count,
        "last_processed": end,
        "max_total_stopping_time": best_tst,
        "max_stopping_time": best_st,
        "max_excursion": best_exc,
    }
