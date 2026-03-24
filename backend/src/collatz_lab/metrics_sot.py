"""Source-of-Truth (SoT) implementations and data contracts for Collatz Lab.

This module is **intentionally self-contained** — it has no imports from other
``collatz_lab`` modules.  It can be loaded in isolation (e.g. in a subprocess,
a property test, or a sandbox) without pulling in the Numba/PyTorch ecosystem.

Design principle (dependency inversion)
---------------------------------------
``services.py`` and all fast backends import *from* this module.  This module
imports from *nobody* in the package.  Validation and SoT checks should anchor
to this module, not to ``services.py``.

Contents
--------
- Shared constants (``INT64_MAX``, ``MAX_KERNEL_STEPS``, …)
- ``NumberMetrics`` and ``AggregateMetrics`` data contracts
- ``collatz_step`` — canonical single Collatz step (Python int)
- ``metrics_direct`` — full orbit, stops at 1, Python bigint (SoT)
- ``metrics_descent_direct`` — descent orbit, stops when below seed, Python bigint (SoT)

What is NOT here
----------------
- Numba JIT kernels, CUDA kernels, MPS/Metal code — those are in ``services.py``
- ``metrics_accelerated`` — an optimized approximation, not a SoT reference
- Aggregate / compute dispatch — ``services.py`` and ``compute.py``
- Validation orchestration — ``validation.py``

Validation protocol
-------------------
See ``docs/CORRECTNESS_AND_VALIDATION.md`` for the full cross-platform protocol.
The differential CI test (``tests/test_differential_cross_backend.py``) validates
all fast backends against ``sieve_reference`` (Python) on every CI run.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants — must stay in lockstep with the JIT kernels in services.py
# ---------------------------------------------------------------------------

INT64_MAX: int = (1 << 63) - 1
COLLATZ_INT64_ODD_STEP_LIMIT: int = (INT64_MAX - 1) // 3
#: Safety step cap.  The longest known Collatz orbit below 2^64 converges well
#: within 2500 steps; 100 000 gives ample headroom without running forever.
MAX_KERNEL_STEPS: int = 100_000


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class NumberMetrics:
    """Collatz orbit metrics for a single seed."""
    total_stopping_time: int
    stopping_time: int
    max_excursion: int


@dataclass
class AggregateMetrics:
    """Aggregate metrics for a range of seeds."""
    processed: int
    last_processed: int
    max_total_stopping_time: dict
    max_stopping_time: dict
    max_excursion: dict
    sample_records: list[dict]


# ---------------------------------------------------------------------------
# Canonical step — used by both SoT reference functions
# ---------------------------------------------------------------------------


def collatz_step(value: int) -> int:
    if value < 1:
        raise ValueError("Collatz is defined only for positive integers.")
    return value // 2 if value % 2 == 0 else (3 * value) + 1


# ---------------------------------------------------------------------------
# Source-of-Truth metric functions (Python int, arbitrary precision)
# ---------------------------------------------------------------------------


def metrics_direct(value: int) -> NumberMetrics:
    """Full Collatz orbit: steps until ``current == 1``.

    Uses Python arbitrary-precision integers — no overflow, no approximation.
    This is the SoT reference for kernels that compute the full orbit
    (``cpu-direct``, ``cpu-accelerated``, ``cpu-parallel``, ``gpu-collatz-accelerated``).
    """
    current = value
    total_steps = 0
    stopping_time: int | None = None
    max_excursion = value
    while current != 1:
        current = collatz_step(current)
        total_steps += 1
        max_excursion = max(max_excursion, current)
        if stopping_time is None and current < value:
            stopping_time = total_steps
    return NumberMetrics(
        total_stopping_time=total_steps,
        stopping_time=stopping_time or total_steps,
        max_excursion=max_excursion,
    )


def metrics_descent_direct(value: int) -> NumberMetrics:
    """Descent Collatz orbit: stops when ``current < seed`` (not at 1).

    Uses Python arbitrary-precision integers — no overflow, no approximation.
    This is the SoT reference for sieve-style kernels (``cpu-sieve``,
    ``gpu-sieve``) which prove convergence by verifying each seed reaches a
    smaller value already covered by ascending-order verification.

    Platform-wide alias: ``collatz_lab.validation_source.metrics_descent_exact``.
    Validation protocol: ``docs/CORRECTNESS_AND_VALIDATION.md``.
    """
    if value < 1:
        raise ValueError("Collatz is defined only for positive integers.")
    if value == 1:
        return NumberMetrics(total_stopping_time=0, stopping_time=0, max_excursion=1)

    current = value
    steps = 0
    max_excursion = value
    while current >= value:
        current = collatz_step(current)
        steps += 1
        max_excursion = max(max_excursion, current)
        if steps >= MAX_KERNEL_STEPS and current >= value:
            raise RuntimeError(
                f"Direct descent verification exceeded MAX_KERNEL_STEPS for seed {value}"
            )

    return NumberMetrics(
        total_stopping_time=steps,
        stopping_time=steps,
        max_excursion=max_excursion,
    )
