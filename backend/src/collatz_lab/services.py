from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import platform
import random as _random
import time
import traceback as _traceback_mod
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .logutil import silence_numba_cuda_info
from .hardware import (
    CPU_ACCELERATED_KERNEL,
    CPU_BARINA_KERNEL,
    CPU_DIRECT_KERNEL,
    CPU_PARALLEL_KERNEL,
    CPU_PARALLEL_ODD_KERNEL,
    CPU_SIEVE_KERNEL,
    GPU_KERNEL,
    GPU_SIEVE_KERNEL,
    cuda_gpu_execution_ready,
    gpu_execution_ready,
)

try:
    import numpy as np
except Exception:  # pragma: no cover - optional GPU dependency
    np = None

try:
    from numba import cuda, njit, prange, set_num_threads
except Exception:  # pragma: no cover - optional GPU dependency
    cuda = None
    njit = None
    prange = None
    set_num_threads = None
from .cpu_sieve_native_runtime import (
    compute_range_metrics_sieve_odd_native,
    cpu_sieve_resolve_backend,
)
from .metrics_sot import (
    AggregateMetrics,
    COLLATZ_INT64_ODD_STEP_LIMIT,
    INT64_MAX,
    MAX_KERNEL_STEPS,
    NumberMetrics,
    collatz_step,
    metrics_descent_direct,
    metrics_direct,
)
from .repository import LabRepository, sha256_text, utc_now
from .schemas import ArtifactKind, ComputeProfile, ModularProbeResult, Run, RunStatus

logger = logging.getLogger("collatz_lab.services")

# ---------------------------------------------------------------------------
# Re-exports from extracted modules (backward-compat for callers and tests)
# ---------------------------------------------------------------------------
from ._profile_helpers import _effective_profile_percent, _positive_int_env  # noqa: E402
from .scheduling import (  # noqa: E402
    LEGACY_VALIDATION_RERUN_OWNER,
    LEGACY_VALIDATION_RERUN_PREFIX,
    OVERFLOW_RECOVERY_OWNER,
    RANDOM_PROBE_OWNER,
    RESEARCH_AUTOPILOT_OWNER,
    _ensure_overflow_recovery_runs,
    _is_legacy_validation_failure,
    _is_overflow_guard_failure,
    _lab_random,
    _prune_duplicate_overflow_recovery_runs,
    _random_probes_enabled,
    _serialized_maintenance_enqueue,
    annotate_legacy_validation_failures,
    queue_continuous_verification_runs,
    queue_legacy_validation_reruns,
    queue_randomized_compute_runs,
    queue_research_snack_runs,
)
from .orchestration import (  # noqa: E402
    _CheckpointWriter,
    _compute_budget_throttle_seconds,
    _effective_checkpoint_interval,
    _run_completion_summary,
    execute_run,
    generate_report,
    probe_modular_claim,
    process_next_queued_run,
)


def accelerated_odd_step(value: int) -> tuple[int, int]:
    if value < 1 or value % 2 == 0:
        raise ValueError("accelerated_odd_step requires a positive odd integer.")
    current = (3 * value) + 1
    shifts = 0
    while current % 2 == 0:
        current //= 2
        shifts += 1
    return current, shifts


GPU_SEEDS_PER_THREAD = 16  # Each thread processes 16 seeds for higher arithmetic intensity

if cuda is not None and np is not None:  # pragma: no branch - import guarded above

    @cuda.jit
    def _collatz_block_metrics_kernel(
        start_value,
        size,
        seeds_per_thread,
        block_total_values,
        block_total_ns,
        block_stopping_values,
        block_stopping_ns,
        block_excursion_values,
        block_excursion_ns,
        block_overflow_ns,
    ):
        tid = cuda.threadIdx.x
        block = cuda.blockIdx.x
        global_tid = block * cuda.blockDim.x + tid

        total_values = cuda.shared.array(256, dtype=np.int32)
        total_ns = cuda.shared.array(256, dtype=np.int64)
        stopping_values = cuda.shared.array(256, dtype=np.int32)
        stopping_ns = cuda.shared.array(256, dtype=np.int64)
        excursion_values = cuda.shared.array(256, dtype=np.int64)
        excursion_ns = cuda.shared.array(256, dtype=np.int64)
        overflow_ns = cuda.shared.array(256, dtype=np.int64)

        # Local per-thread best across all seeds this thread handles
        best_tst = -1
        best_tst_n = start_value
        best_st = -1
        best_st_n = start_value
        best_exc = np.int64(-1)
        best_exc_n = start_value
        first_overflow = np.int64(0)

        base_index = global_tid * seeds_per_thread
        for s in range(seeds_per_thread):
            index = base_index + s
            if index >= size:
                break

            original = start_value + index
            current = original
            steps = 0
            stopping_time = -1
            max_excursion = current

            while current != 1 and steps < MAX_KERNEL_STEPS:
                if current & 1 == 0:
                    current >>= 1
                    steps += 1
                    if stopping_time < 0 and current < original:
                        stopping_time = steps
                else:
                    if current > COLLATZ_INT64_ODD_STEP_LIMIT:
                        if first_overflow == 0:
                            first_overflow = original
                        steps = -1
                        stopping_time = -1
                        max_excursion = -1
                        break
                    current = 3 * current + 1
                    steps += 1
                    if current > max_excursion:
                        max_excursion = current
                    if stopping_time < 0 and current < original:
                        stopping_time = steps
                    while current & 1 == 0:
                        current >>= 1
                        steps += 1
                        if stopping_time < 0 and current < original:
                            stopping_time = steps
                if current > max_excursion:
                    max_excursion = current

            if current != 1 and steps >= MAX_KERNEL_STEPS:
                if first_overflow == 0:
                    first_overflow = original
                steps = -1
                stopping_time = -1
                max_excursion = -1

            if stopping_time < 0 and steps >= 0:
                stopping_time = steps

            # Update per-thread best
            if steps > best_tst or (steps == best_tst and original < best_tst_n):
                best_tst = steps
                best_tst_n = original
            if stopping_time > best_st or (stopping_time == best_st and original < best_st_n):
                best_st = stopping_time
                best_st_n = original
            if max_excursion > best_exc or (max_excursion == best_exc and original < best_exc_n):
                best_exc = max_excursion
                best_exc_n = original

        total_values[tid] = best_tst
        total_ns[tid] = best_tst_n
        stopping_values[tid] = best_st
        stopping_ns[tid] = best_st_n
        excursion_values[tid] = best_exc
        excursion_ns[tid] = best_exc_n
        overflow_ns[tid] = first_overflow
        cuda.syncthreads()

        stride = cuda.blockDim.x // 2
        while stride > 0:
            if tid < stride:
                other = tid + stride

                if (
                    total_values[other] > total_values[tid]
                    or (
                        total_values[other] == total_values[tid]
                        and total_ns[other] < total_ns[tid]
                    )
                ):
                    total_values[tid] = total_values[other]
                    total_ns[tid] = total_ns[other]

                if (
                    stopping_values[other] > stopping_values[tid]
                    or (
                        stopping_values[other] == stopping_values[tid]
                        and stopping_ns[other] < stopping_ns[tid]
                    )
                ):
                    stopping_values[tid] = stopping_values[other]
                    stopping_ns[tid] = stopping_ns[other]

                if (
                    excursion_values[other] > excursion_values[tid]
                    or (
                        excursion_values[other] == excursion_values[tid]
                        and excursion_ns[other] < excursion_ns[tid]
                    )
                ):
                    excursion_values[tid] = excursion_values[other]
                    excursion_ns[tid] = excursion_ns[other]

                if overflow_ns[tid] == 0 or (
                    overflow_ns[other] > 0 and overflow_ns[other] < overflow_ns[tid]
                ):
                    overflow_ns[tid] = overflow_ns[other]
            cuda.syncthreads()
            stride //= 2

        if tid == 0:
            block_total_values[block] = total_values[0]
            block_total_ns[block] = total_ns[0]
            block_stopping_values[block] = stopping_values[0]
            block_stopping_ns[block] = stopping_ns[0]
            block_excursion_values[block] = excursion_values[0]
            block_excursion_ns[block] = excursion_ns[0]
            block_overflow_ns[block] = overflow_ns[0]


if np is not None and njit is not None:  # pragma: no branch - import guarded above

    @njit(cache=True, parallel=True)
    def _collatz_metrics_parallel(start_value, size):
        total_steps = np.empty(size, dtype=np.int32)
        stopping_steps = np.empty(size, dtype=np.int32)
        max_excursions = np.empty(size, dtype=np.int64)

        for index in prange(size):
            current = np.int64(start_value + index)
            original = current
            steps = np.int32(0)
            stopping_time = np.int32(-1)
            max_excursion = current

            while current != 1 and steps < MAX_KERNEL_STEPS:
                if current & 1 == 0:
                    current >>= 1
                    steps += 1
                    if stopping_time < 0 and current < original:
                        stopping_time = steps
                else:
                    if current > COLLATZ_INT64_ODD_STEP_LIMIT:
                        steps = np.int32(-1)
                        stopping_time = np.int32(-1)
                        max_excursion = np.int64(-1)
                        break
                    current = 3 * current + 1
                    steps += 1
                    if current > max_excursion:
                        max_excursion = current
                    if stopping_time < 0 and current < original:
                        stopping_time = steps
                    while current & 1 == 0:
                        current >>= 1
                        steps += 1
                        if stopping_time < 0 and current < original:
                            stopping_time = steps
                if current > max_excursion:
                    max_excursion = current

            if current != 1 and steps >= MAX_KERNEL_STEPS:
                steps = np.int32(-1)
                stopping_time = np.int32(-1)
                max_excursion = np.int64(-1)

            total_steps[index] = steps
            stopping_steps[index] = stopping_time if stopping_time >= 0 else steps
            max_excursions[index] = max_excursion

        return total_steps, stopping_steps, max_excursions

    @njit(cache=True, parallel=True)
    def _collatz_metrics_parallel_odd(first_odd, odd_count):
        """Process only odd seeds: first_odd, first_odd+2, first_odd+4, ...

        Mathematical justification: for any even n, n → n/2 < n, so
        convergence of all odd seeds ≤ N implies convergence of all
        integers ≤ N by strong induction on v2(n).

        Returns per-seed metrics arrays indexed by odd-seed position.
        """
        total_steps = np.empty(odd_count, dtype=np.int32)
        stopping_steps = np.empty(odd_count, dtype=np.int32)
        max_excursions = np.empty(odd_count, dtype=np.int64)

        for index in prange(odd_count):
            current = np.int64(first_odd + 2 * index)
            original = current
            steps = np.int32(0)
            stopping_time = np.int32(-1)
            max_excursion = current

            while current != 1 and steps < MAX_KERNEL_STEPS:
                if current & 1 == 0:
                    current >>= 1
                    steps += 1
                    if stopping_time < 0 and current < original:
                        stopping_time = steps
                else:
                    if current > COLLATZ_INT64_ODD_STEP_LIMIT:
                        steps = np.int32(-1)
                        stopping_time = np.int32(-1)
                        max_excursion = np.int64(-1)
                        break
                    current = 3 * current + 1
                    steps += 1
                    if current > max_excursion:
                        max_excursion = current
                    if stopping_time < 0 and current < original:
                        stopping_time = steps
                    while current & 1 == 0:
                        current >>= 1
                        steps += 1
                        if stopping_time < 0 and current < original:
                            stopping_time = steps
                if current > max_excursion:
                    max_excursion = current

            if current != 1 and steps >= MAX_KERNEL_STEPS:
                steps = np.int32(-1)
                stopping_time = np.int32(-1)
                max_excursion = np.int64(-1)

            total_steps[index] = steps
            stopping_steps[index] = stopping_time if stopping_time >= 0 else steps
            max_excursions[index] = max_excursion

        return total_steps, stopping_steps, max_excursions

    @njit(cache=True, parallel=True)
    def _collatz_metrics_parallel_descent(first_value, size):
        """Standard Collatz until first descent below the original seed."""
        total_steps = np.empty(size, dtype=np.int32)
        stopping_steps = np.empty(size, dtype=np.int32)
        max_excursions = np.empty(size, dtype=np.int64)

        for index in prange(size):
            current = np.int64(first_value + index)
            original = current

            if original <= 1:
                total_steps[index] = np.int32(0)
                stopping_steps[index] = np.int32(0)
                max_excursions[index] = original
                continue

            steps = np.int32(0)
            max_excursion = current

            while current >= original and steps < MAX_KERNEL_STEPS:
                if current & 1 == 0:
                    current >>= 1
                    steps += 1
                else:
                    if current > COLLATZ_INT64_ODD_STEP_LIMIT:
                        steps = np.int32(-1)
                        max_excursion = np.int64(-1)
                        break
                    current = 3 * current + 1
                    steps += 1
                    if current > max_excursion:
                        max_excursion = current
                    while current & 1 == 0 and current >= original:
                        current >>= 1
                        steps += 1

                if current > max_excursion:
                    max_excursion = current

            if current >= original and steps >= MAX_KERNEL_STEPS:
                steps = np.int32(-1)
                max_excursion = np.int64(-1)

            total_steps[index] = steps
            stopping_steps[index] = steps
            max_excursions[index] = max_excursion

        return total_steps, stopping_steps, max_excursions

    @njit(cache=True, parallel=True)
    def _collatz_metrics_parallel_descent_odd(first_odd, odd_count):
        """Odd-only standard Collatz until first descent below the seed."""
        total_steps = np.empty(odd_count, dtype=np.int32)
        stopping_steps = np.empty(odd_count, dtype=np.int32)
        max_excursions = np.empty(odd_count, dtype=np.int64)

        for index in prange(odd_count):
            current = np.int64(first_odd + 2 * index)
            original = current

            if original <= 1:
                total_steps[index] = np.int32(0)
                stopping_steps[index] = np.int32(0)
                max_excursions[index] = original
                continue

            steps = np.int32(0)
            max_excursion = current

            while current >= original and steps < MAX_KERNEL_STEPS:
                if current & 1 == 0:
                    current >>= 1
                    steps += 1
                else:
                    if current > COLLATZ_INT64_ODD_STEP_LIMIT:
                        steps = np.int32(-1)
                        max_excursion = np.int64(-1)
                        break
                    current = 3 * current + 1
                    steps += 1
                    if current > max_excursion:
                        max_excursion = current
                    while current & 1 == 0 and current >= original:
                        current >>= 1
                        steps += 1

                if current > max_excursion:
                    max_excursion = current

            if current >= original and steps >= MAX_KERNEL_STEPS:
                steps = np.int32(-1)
                max_excursion = np.int64(-1)

            total_steps[index] = steps
            stopping_steps[index] = steps
            max_excursions[index] = max_excursion

        return total_steps, stopping_steps, max_excursions


# ── Sieve shortcut tables ─────────────────────────────────────────
# Process SIEVE_K "combined Collatz steps" per table lookup.
# A combined step: if odd → (3n+1)/2, if even → n/2.
# After k combined steps on n = 2^k * q + b:
#   result = SIEVE_MUL[b] * q + SIEVE_OFF[b]
# This replaces k individual steps with one multiply + add.

SIEVE_K = 16
_SIEVE_SIZE = 1 << SIEVE_K
_SIEVE_MASK = _SIEVE_SIZE - 1


def _build_sieve_tables(k: int = SIEVE_K):
    """Build shortcut lookup tables for k-bit combined Collatz steps.

    Returns (multipliers, offsets, odd_counts, safe_limits) as numpy int64/int32 arrays.
    """
    size = 1 << k
    multipliers = np.empty(size, dtype=np.int64)
    offsets = np.empty(size, dtype=np.int64)
    odd_counts = np.empty(size, dtype=np.int32)
    safe_limits = np.empty(size, dtype=np.int64)

    for b in range(size):
        # Simulate k combined steps on b (q=0 case)
        val_lo = b
        odds = 0
        for _ in range(k):
            if val_lo & 1:
                val_lo = (3 * val_lo + 1) >> 1
                odds += 1
            else:
                val_lo >>= 1

        # Simulate k combined steps on b + 2^k (q=1 case)
        val_hi = b + size
        for _ in range(k):
            if val_hi & 1:
                val_hi = (3 * val_hi + 1) >> 1
            else:
                val_hi >>= 1

        mul = val_hi - val_lo
        offsets[b] = val_lo
        multipliers[b] = mul
        odd_counts[b] = odds
        safe_limits[b] = INT64_MAX // mul if mul > 0 else INT64_MAX

    return multipliers, offsets, odd_counts, safe_limits


_sieve_cache: tuple | None = None


def _get_sieve_tables():
    global _sieve_cache
    if _sieve_cache is None:
        _sieve_cache = _build_sieve_tables()
    return _sieve_cache


# ── Powers-of-3 table for Barina's domain-switching algorithm ──────
# Only need entries up to the max possible CTZ of an int64 value (63).
_POW3_SIZE = 64
_pow3_cache = None


def _get_pow3_table():
    global _pow3_cache
    if _pow3_cache is None:
        _pow3_cache = np.empty(_POW3_SIZE, dtype=np.int64)
        val = 1
        for i in range(_POW3_SIZE):
            _pow3_cache[i] = val
            if val <= INT64_MAX // 3:
                val *= 3
            else:
                val = INT64_MAX  # overflow sentinel
    return _pow3_cache


if np is not None and njit is not None:

    @njit(cache=True)
    def _ctz64(n):
        """Count trailing zeros of a 64-bit integer."""
        if n == 0:
            return 64
        count = np.int32(0)
        val = n
        # Binary search for trailing zeros (branchless-ish)
        if (val & 0xFFFFFFFF) == 0:
            count += 32
            val >>= 32
        if (val & 0xFFFF) == 0:
            count += 16
            val >>= 16
        if (val & 0xFF) == 0:
            count += 8
            val >>= 8
        if (val & 0xF) == 0:
            count += 4
            val >>= 4
        if (val & 0x3) == 0:
            count += 2
            val >>= 2
        if (val & 0x1) == 0:
            count += 1
        return count

    @njit(cache=True, parallel=True)
    def _collatz_sieve_parallel_odd(
        first_odd, odd_count, multipliers, offsets, odd_counts, safe_limits, k
    ):
        """Descent-verification kernel using standard Collatz with early termination.

        This kernel proves convergence by induction:
        - Processes ALL odd seeds in [first_odd, first_odd + 2*(odd_count-1)]
        - Stops when orbit drops below seed (descent verification)
        - Returns descent time as stopping_time (exact step count)
        - total_stopping_time = stopping_time (descent only, NOT full orbit to 1)
        - max_excursion = peak value during descent (NOT full orbit peak)

        Every seed is individually verified using standard Collatz steps
        (3n+1 / n/2).  No seeds are skipped or filtered.

        The shortcut-table parameters are accepted for interface compatibility
        but are not used — step-by-step is faster for early termination because
        the table would overshoot the descent point.
        """
        total_steps_out = np.empty(odd_count, dtype=np.int32)
        stopping_steps_out = np.empty(odd_count, dtype=np.int32)
        max_excursions_out = np.empty(odd_count, dtype=np.int64)

        for index in prange(odd_count):
            seed = np.int64(first_odd + 2 * index)

            # seed=1 is trivially converged (cycle 1→4→2→1).
            if seed <= 1:
                total_steps_out[index] = np.int32(0)
                stopping_steps_out[index] = np.int32(0)
                max_excursions_out[index] = seed
                continue

            current = seed
            steps = np.int32(0)
            max_excursion = current

            while current >= seed and steps < MAX_KERNEL_STEPS:
                if current & 1 == 0:
                    current >>= 1
                    steps += 1
                else:
                    if current > COLLATZ_INT64_ODD_STEP_LIMIT:
                        steps = np.int32(-1)
                        max_excursion = np.int64(-1)
                        break
                    current = 3 * current + 1
                    steps += 1
                    if current > max_excursion:
                        max_excursion = current
                    while current & 1 == 0 and current >= seed:
                        current >>= 1
                        steps += 1

            if current >= seed and current != 1 and steps >= MAX_KERNEL_STEPS:
                steps = np.int32(-1)
                max_excursion = np.int64(-1)

            total_steps_out[index] = steps
            stopping_steps_out[index] = steps
            max_excursions_out[index] = max_excursion

        return total_steps_out, stopping_steps_out, max_excursions_out

    @njit(cache=True, parallel=True)
    def _collatz_barina_parallel_odd(first_odd, odd_count, pow3_table):
        """Barina domain-switching kernel with early termination.

        Based on Barina 2020 (verified Collatz to 2^71).  Replaces the
        standard if-odd/if-even branching with a domain-switching loop
        using CTZ + precomputed powers of 3.

        The transformation is mathematically equivalent to standard Collatz
        iteration but groups multiple steps into two CTZ+shift+multiply
        operations per loop iteration:

            while n >= n0:
                n += 1;  alpha = ctz(n);  n = (n >> alpha) * 3^alpha
                n -= 1;  beta  = ctz(n);  n >>= beta

        IMPORTANT: The step count reported by this kernel is NOT the same as
        standard Collatz step count.  Each loop iteration processes
        (alpha + beta) compressed steps.  The convergence proof (descent
        below seed) is equivalent, but the step metric differs.

        This kernel is EXPERIMENTAL until validated against standard Collatz
        for the full range of interest.  Every seed is individually verified;
        no seeds are skipped or filtered.
        """
        stopping_steps_out = np.empty(odd_count, dtype=np.int32)
        max_excursions_out = np.empty(odd_count, dtype=np.int64)

        for index in prange(odd_count):
            seed = np.int64(first_odd + 2 * index)

            # seed=1 is trivially converged.
            if seed <= 1:
                stopping_steps_out[index] = np.int32(0)
                max_excursions_out[index] = seed
                continue

            n = seed
            steps = np.int32(0)
            max_excursion = n

            while n >= seed and steps < MAX_KERNEL_STEPS:
                # Domain switch: make even by adding 1
                n += 1
                alpha = _ctz64(n)
                n >>= alpha

                # Check overflow before multiply by 3^alpha
                if alpha < 64 and n > INT64_MAX // pow3_table[alpha]:
                    # Overflow: fall back to marking as overflow
                    steps = np.int32(-1)
                    max_excursion = np.int64(-1)
                    break

                n *= pow3_table[alpha]
                steps += alpha  # alpha even-steps compressed

                # Switch back
                n -= 1
                if n > max_excursion:
                    max_excursion = n

                beta = _ctz64(n)
                n >>= beta
                steps += beta  # beta more even-steps

            if n >= seed and n != 1 and steps >= MAX_KERNEL_STEPS:
                steps = np.int32(-1)
                max_excursion = np.int64(-1)

            stopping_steps_out[index] = steps
            max_excursions_out[index] = max_excursion

        return stopping_steps_out, max_excursions_out


if cuda is not None and np is not None:

    @cuda.jit
    def _collatz_sieve_gpu_kernel(
        start_value,
        size,
        seeds_per_thread,
        k,
        d_multipliers,
        d_offsets,
        d_odd_counts,
        d_safe_limits,
        block_total_values,
        block_total_ns,
        block_stopping_values,
        block_stopping_ns,
        block_excursion_values,
        block_excursion_ns,
        block_overflow_ns,
    ):
        """GPU sieve kernel with early termination.

        Stops when current < original (convergence proven by induction).
        Pure step-by-step iteration for minimum per-seed overhead.
        """
        tid = cuda.threadIdx.x
        block = cuda.blockIdx.x
        global_tid = block * cuda.blockDim.x + tid

        total_vals = cuda.shared.array(256, dtype=np.int32)
        total_ns = cuda.shared.array(256, dtype=np.int64)
        stopping_vals = cuda.shared.array(256, dtype=np.int32)
        stopping_ns = cuda.shared.array(256, dtype=np.int64)
        excursion_vals = cuda.shared.array(256, dtype=np.int64)
        excursion_ns = cuda.shared.array(256, dtype=np.int64)
        overflow_ns = cuda.shared.array(256, dtype=np.int64)

        best_tst = -1
        best_tst_n = start_value
        best_st = -1
        best_st_n = start_value
        best_exc = np.int64(-1)
        best_exc_n = start_value
        first_overflow = np.int64(0)

        base_index = global_tid * seeds_per_thread
        for s in range(seeds_per_thread):
            idx = base_index + s
            if idx >= size:
                break

            # Odd-only: map linear index to odd seed (2*idx offset from first odd)
            original = start_value + 2 * idx
            if original <= 1:
                if original < best_tst_n or best_tst == -1:
                    best_tst = 0
                    best_tst_n = original
                    best_st = 0
                    best_st_n = original
                    best_exc = original
                    best_exc_n = original
                continue
            current = original
            steps = 0
            max_excursion = current

            while current >= original and steps < MAX_KERNEL_STEPS:
                if current & 1 == 0:
                    current >>= 1
                    steps += 1
                else:
                    if current > COLLATZ_INT64_ODD_STEP_LIMIT:
                        if first_overflow == 0:
                            first_overflow = original
                        steps = -1
                        max_excursion = -1
                        break
                    current = 3 * current + 1
                    steps += 1
                    if current > max_excursion:
                        max_excursion = current
                    while current & 1 == 0 and current >= original:
                        current >>= 1
                        steps += 1

            if current >= original and steps >= MAX_KERNEL_STEPS:
                if first_overflow == 0:
                    first_overflow = original
                steps = -1
                max_excursion = -1

            if steps > best_tst or (steps == best_tst and original < best_tst_n):
                best_tst = steps
                best_tst_n = original
            if steps > best_st or (steps == best_st and original < best_st_n):
                best_st = steps
                best_st_n = original
            if max_excursion > best_exc or (max_excursion == best_exc and original < best_exc_n):
                best_exc = max_excursion
                best_exc_n = original

        total_vals[tid] = best_tst
        total_ns[tid] = best_tst_n
        stopping_vals[tid] = best_st
        stopping_ns[tid] = best_st_n
        excursion_vals[tid] = best_exc
        excursion_ns[tid] = best_exc_n
        overflow_ns[tid] = first_overflow
        cuda.syncthreads()

        stride = cuda.blockDim.x // 2
        while stride > 0:
            if tid < stride:
                other = tid + stride
                if (
                    total_vals[other] > total_vals[tid]
                    or (total_vals[other] == total_vals[tid] and total_ns[other] < total_ns[tid])
                ):
                    total_vals[tid] = total_vals[other]
                    total_ns[tid] = total_ns[other]
                if (
                    stopping_vals[other] > stopping_vals[tid]
                    or (stopping_vals[other] == stopping_vals[tid] and stopping_ns[other] < stopping_ns[tid])
                ):
                    stopping_vals[tid] = stopping_vals[other]
                    stopping_ns[tid] = stopping_ns[other]
                if (
                    excursion_vals[other] > excursion_vals[tid]
                    or (excursion_vals[other] == excursion_vals[tid] and excursion_ns[other] < excursion_ns[tid])
                ):
                    excursion_vals[tid] = excursion_vals[other]
                    excursion_ns[tid] = excursion_ns[other]
                if overflow_ns[tid] == 0 or (
                    overflow_ns[other] > 0 and overflow_ns[other] < overflow_ns[tid]
                ):
                    overflow_ns[tid] = overflow_ns[other]
            cuda.syncthreads()
            stride //= 2

        if tid == 0:
            block_total_values[block] = total_vals[0]
            block_total_ns[block] = total_ns[0]
            block_stopping_values[block] = stopping_vals[0]
            block_stopping_ns[block] = stopping_ns[0]
            block_excursion_values[block] = excursion_vals[0]
            block_excursion_ns[block] = excursion_ns[0]
            block_overflow_ns[block] = overflow_ns[0]



def metrics_accelerated(value: int) -> NumberMetrics:
    current = value
    total_steps = 0
    stopping_time: int | None = None
    max_excursion = value
    while current != 1:
        if current % 2 == 0:
            current //= 2
            total_steps += 1
            if stopping_time is None and current < value:
                stopping_time = total_steps
        else:
            current = (3 * current) + 1
            total_steps += 1
            max_excursion = max(max_excursion, current)
            if stopping_time is None and current < value:
                stopping_time = total_steps
            while current % 2 == 0:
                current //= 2
                total_steps += 1
                if stopping_time is None and current < value:
                    stopping_time = total_steps
        max_excursion = max(max_excursion, current)
    return NumberMetrics(
        total_stopping_time=total_steps,
        stopping_time=stopping_time or total_steps,
        max_excursion=max_excursion,
    )



def _aggregate_metrics_from_arrays(
    first_value: int,
    total_steps,
    stopping_steps,
    max_excursions,
    *,
    sample_limit: int,
) -> AggregateMetrics:
    count = len(total_steps)
    max_total = {"n": first_value, "value": -1}
    max_stopping = {"n": first_value, "value": -1}
    max_excursion = {"n": first_value, "value": -1}
    sample_records: list[dict] = []

    for index in range(count):
        value = first_value + index
        total = int(total_steps[index])
        stopping = int(stopping_steps[index])
        excursion = int(max_excursions[index])

        if total > max_total["value"]:
            max_total = {"n": value, "value": total}
            if len(sample_records) < sample_limit:
                sample_records.append({"metric": "max_total_stopping_time", **max_total})

        if stopping > max_stopping["value"]:
            max_stopping = {"n": value, "value": stopping}
            if len(sample_records) < sample_limit:
                sample_records.append({"metric": "max_stopping_time", **max_stopping})

        if excursion > max_excursion["value"]:
            max_excursion = {"n": value, "value": excursion}
            if len(sample_records) < sample_limit:
                sample_records.append({"metric": "max_excursion", **max_excursion})

    return AggregateMetrics(
        processed=count,
        last_processed=first_value + count - 1,
        max_total_stopping_time=max_total,
        max_stopping_time=max_stopping,
        max_excursion=max_excursion,
        sample_records=sample_records,
    )


def _pick_better_metric(current: dict, candidate: dict) -> dict:
    if candidate["value"] > current["value"]:
        return candidate
    if candidate["value"] == current["value"] and candidate["n"] < current["n"]:
        return candidate
    return current


def _aggregate_metrics_from_block_summaries(
    first_value: int,
    count: int,
    block_total_values,
    block_total_ns,
    block_stopping_values,
    block_stopping_ns,
    block_excursion_values,
    block_excursion_ns,
    *,
    sample_limit: int,
) -> AggregateMetrics:
    max_total = {"n": first_value, "value": -1}
    max_stopping = {"n": first_value, "value": -1}
    max_excursion = {"n": first_value, "value": -1}
    sample_records: list[dict] = []

    block_count = len(block_total_values)
    for index in range(block_count):
        total_candidate = {
            "n": int(block_total_ns[index]),
            "value": int(block_total_values[index]),
        }
        stopping_candidate = {
            "n": int(block_stopping_ns[index]),
            "value": int(block_stopping_values[index]),
        }
        excursion_candidate = {
            "n": int(block_excursion_ns[index]),
            "value": int(block_excursion_values[index]),
        }

        next_total = _pick_better_metric(max_total, total_candidate)
        if next_total is not max_total and len(sample_records) < sample_limit:
            sample_records.append({"metric": "max_total_stopping_time", **next_total})
        max_total = next_total

        next_stopping = _pick_better_metric(max_stopping, stopping_candidate)
        if next_stopping is not max_stopping and len(sample_records) < sample_limit:
            sample_records.append({"metric": "max_stopping_time", **next_stopping})
        max_stopping = next_stopping

        next_excursion = _pick_better_metric(max_excursion, excursion_candidate)
        if next_excursion is not max_excursion and len(sample_records) < sample_limit:
            sample_records.append({"metric": "max_excursion", **next_excursion})
        max_excursion = next_excursion

    return AggregateMetrics(
        processed=count,
        last_processed=first_value + count - 1,
        max_total_stopping_time=max_total,
        max_stopping_time=max_stopping,
        max_excursion=max_excursion,
        sample_records=sample_records,
    )


def _gpu_threads_per_block(profile: ComputeProfile | None = None) -> int:
    effective_percent = _effective_profile_percent(profile, "gpu")
    default_value = 128 if effective_percent <= 25 else 256
    value = _positive_int_env("COLLATZ_GPU_THREADS_PER_BLOCK", default_value)
    # Capped at 256 — shared memory arrays in the kernel are sized 256.
    # With 256 threads/block: ~11KB shared mem → 4 blocks/SM on RTX 4060 Ti
    # (vs 1 block/SM at 1024), giving much better latency hiding and occupancy.
    return max(64, min(256, value))


def _mps_accelerated_chunk_size() -> int:
    """Larger chunks → fewer MPS syncs; Apple Silicon unified memory allows big batches."""
    return max(1024, min(2_097_152, _positive_int_env("COLLATZ_MPS_BATCH_SIZE", 32_768)))


def _mps_sieve_chunk_size() -> int:
    # Default 1M odds per MPS sub-batch: ``scripts/profile_mps_metal_sieve.py --quick`` median winner
    # on Apple Silicon (beats 2M/128; sync 256 regressed in the same sweep). Still clamped 4k…4M.
    # Tiny values (e.g. 131k) caused thousands of Python iterations per huge checkpoint.
    return max(4096, min(4_194_304, _positive_int_env("COLLATZ_MPS_SIEVE_BATCH_SIZE", 1_048_576)))


def _compute_range_metrics_gpu_mps(
    start: int,
    end: int,
    *,
    start_at: int | None = None,
    sample_limit: int = 12,
    profile: ComputeProfile | None = None,
) -> AggregateMetrics:
    """Apple MPS (PyTorch) — same metrics contract as CUDA accelerated kernel."""
    from . import mps_collatz

    if np is None:
        raise ValueError("numpy is required for MPS GPU execution.")
    if start < 1 or end < start:
        raise ValueError("Invalid run interval.")
    first = start_at if start_at is not None else start
    if first < start or first > end:
        raise ValueError("Checkpoint start is outside the run interval.")
    if not mps_collatz.mps_accelerated_available():
        raise ValueError("MPS is not available.")

    size = end - first + 1
    total_steps, stopping_steps, max_excursions = mps_collatz.compute_accelerated_arrays_mps(
        first, size, batch_size=_mps_accelerated_chunk_size()
    )
    patches = _fallback_overflow_seeds(first, total_steps, stopping_steps, max_excursions)
    aggregate = _aggregate_metrics_from_arrays(
        first,
        total_steps,
        stopping_steps,
        max_excursions,
        sample_limit=sample_limit,
    )
    for patch in patches:
        if patch.max_excursion > aggregate.max_excursion["value"]:
            aggregate.max_excursion = {"n": patch.seed, "value": patch.max_excursion}
        if patch.total_stopping_time > aggregate.max_total_stopping_time["value"]:
            aggregate.max_total_stopping_time = {"n": patch.seed, "value": patch.total_stopping_time}
        if patch.stopping_time > aggregate.max_stopping_time["value"]:
            aggregate.max_stopping_time = {"n": patch.seed, "value": patch.stopping_time}
    return aggregate


def compute_range_metrics_gpu(
    start: int,
    end: int,
    *,
    start_at: int | None = None,
    sample_limit: int = 12,
    profile: ComputeProfile | None = None,
) -> AggregateMetrics:
    if not gpu_execution_ready() or np is None:
        raise ValueError("GPU execution is not available on this machine.")
    silence_numba_cuda_info()
    if start < 1 or end < start:
        raise ValueError("Invalid run interval.")
    first = start_at if start_at is not None else start
    if first < start or first > end:
        raise ValueError("Checkpoint start is outside the run interval.")

    if cuda is not None and cuda_gpu_execution_ready():
        size = end - first + 1
        threads_per_block = _gpu_threads_per_block(profile)
        seeds_per_thread = GPU_SEEDS_PER_THREAD
        total_threads = math.ceil(size / seeds_per_thread)
        blocks_per_grid = math.ceil(total_threads / threads_per_block)
        device_block_total_values = cuda.device_array(blocks_per_grid, dtype=np.int32)
        device_block_total_ns = cuda.device_array(blocks_per_grid, dtype=np.int64)
        device_block_stopping_values = cuda.device_array(blocks_per_grid, dtype=np.int32)
        device_block_stopping_ns = cuda.device_array(blocks_per_grid, dtype=np.int64)
        device_block_excursion_values = cuda.device_array(blocks_per_grid, dtype=np.int64)
        device_block_excursion_ns = cuda.device_array(blocks_per_grid, dtype=np.int64)
        device_block_overflow_ns = cuda.device_array(blocks_per_grid, dtype=np.int64)

        _collatz_block_metrics_kernel[blocks_per_grid, threads_per_block](
            first,
            size,
            seeds_per_thread,
            device_block_total_values,
            device_block_total_ns,
            device_block_stopping_values,
            device_block_stopping_ns,
            device_block_excursion_values,
            device_block_excursion_ns,
            device_block_overflow_ns,
        )
        cuda.synchronize()

        block_overflow_ns = device_block_overflow_ns.copy_to_host()
        overflow_hits = [int(value) for value in block_overflow_ns if int(value) > 0]

        try:
            deallocations = getattr(cuda.current_context(), "deallocations", None)
            if deallocations is not None:
                deallocations.clear()
        except Exception:
            pass

        if overflow_hits:
            return compute_range_metrics_parallel(
                start, end, start_at=first, sample_limit=sample_limit, profile=profile,
            )

        return _aggregate_metrics_from_block_summaries(
            first,
            size,
            device_block_total_values.copy_to_host(),
            device_block_total_ns.copy_to_host(),
            device_block_stopping_values.copy_to_host(),
            device_block_stopping_ns.copy_to_host(),
            device_block_excursion_values.copy_to_host(),
            device_block_excursion_ns.copy_to_host(),
            sample_limit=sample_limit,
        )

    from . import mps_collatz

    if mps_collatz.mps_accelerated_available():
        return _compute_range_metrics_gpu_mps(
            start,
            end,
            start_at=start_at,
            sample_limit=sample_limit,
            profile=profile,
        )

    raise ValueError("GPU execution is not available on this machine.")


@dataclass
class _OverflowPatch:
    """Holds true Python-int metrics for seeds whose max_excursion
    overflows int64, so the aggregate can use the real values."""
    seed: int
    total_stopping_time: int
    stopping_time: int
    max_excursion: int


def _fallback_overflow_seeds(
    first: int,
    total_steps,
    stopping_steps,
    max_excursions,
) -> list[_OverflowPatch]:
    """Process overflow seeds (marked as -1 by Numba) using Python arbitrary
    precision arithmetic.  Mutates the arrays in-place and returns patches
    for seeds whose excursion exceeds int64."""
    overflow_indices = np.where(total_steps < 0)[0]
    if overflow_indices.size == 0:
        return []
    patches: list[_OverflowPatch] = []
    for idx in overflow_indices:
        seed = first + int(idx)
        m = metrics_accelerated(seed)
        total_steps[idx] = m.total_stopping_time
        stopping_steps[idx] = m.stopping_time
        if m.max_excursion > INT64_MAX:
            max_excursions[idx] = INT64_MAX  # placeholder
            patches.append(_OverflowPatch(seed, m.total_stopping_time, m.stopping_time, m.max_excursion))
        else:
            max_excursions[idx] = m.max_excursion
    return patches


def _fallback_overflow_seeds_descent(
    first: int,
    total_steps,
    stopping_steps,
    max_excursions,
    *,
    stride: int = 1,
) -> list[_OverflowPatch]:
    """Recover descent-style metrics for overflow seeds using Python bigint."""
    overflow_indices = np.where(total_steps < 0)[0]
    if overflow_indices.size == 0:
        return []
    patches: list[_OverflowPatch] = []
    for idx in overflow_indices:
        seed = first + (stride * int(idx))
        m = metrics_descent_direct(seed)
        total_steps[idx] = m.total_stopping_time
        stopping_steps[idx] = m.stopping_time
        if m.max_excursion > INT64_MAX:
            max_excursions[idx] = INT64_MAX
            patches.append(_OverflowPatch(seed, m.total_stopping_time, m.stopping_time, m.max_excursion))
        else:
            max_excursions[idx] = m.max_excursion
    return patches


def _aggregate_metrics_from_strided_arrays(
    first_value: int,
    stride: int,
    total_steps,
    stopping_steps,
    max_excursions,
    *,
    sample_limit: int,
) -> AggregateMetrics:
    count = len(total_steps)
    max_total = {"n": first_value, "value": -1}
    max_stopping = {"n": first_value, "value": -1}
    max_excursion = {"n": first_value, "value": -1}
    sample_records: list[dict] = []

    for index in range(count):
        value = first_value + (stride * index)
        total = int(total_steps[index])
        stopping = int(stopping_steps[index])
        excursion = int(max_excursions[index])

        if total > max_total["value"]:
            max_total = {"n": value, "value": total}
            if len(sample_records) < sample_limit:
                sample_records.append({"metric": "max_total_stopping_time", **max_total})

        if stopping > max_stopping["value"]:
            max_stopping = {"n": value, "value": stopping}
            if len(sample_records) < sample_limit:
                sample_records.append({"metric": "max_stopping_time", **max_stopping})

        if excursion > max_excursion["value"]:
            max_excursion = {"n": value, "value": excursion}
            if len(sample_records) < sample_limit:
                sample_records.append({"metric": "max_excursion", **max_excursion})

    return AggregateMetrics(
        processed=count,
        last_processed=first_value + (stride * (count - 1) if count else 0),
        max_total_stopping_time=max_total,
        max_stopping_time=max_stopping,
        max_excursion=max_excursion,
        sample_records=sample_records,
    )


def compute_range_metrics_parallel(
    start: int,
    end: int,
    *,
    start_at: int | None = None,
    sample_limit: int = 12,
    profile: ComputeProfile | None = None,
) -> AggregateMetrics:
    if np is None or njit is None or prange is None:
        raise ValueError("CPU parallel execution requires numpy and numba.")
    if start < 1 or end < start:
        raise ValueError("Invalid run interval.")
    first = start_at if start_at is not None else start
    if first < start or first > end:
        raise ValueError("Checkpoint start is outside the run interval.")

    if set_num_threads is not None:
        total_cores = os.cpu_count() or 1
        effective_percent = _effective_profile_percent(profile, "cpu")
        thread_count = max(1, min(total_cores, round(total_cores * max(effective_percent, 1) / 100)))
        set_num_threads(thread_count)

    size = end - first + 1
    total_steps, stopping_steps, max_excursions = _collatz_metrics_parallel(first, size)
    patches = _fallback_overflow_seeds(first, total_steps, stopping_steps, max_excursions)
    aggregate = _aggregate_metrics_from_arrays(
        first,
        total_steps,
        stopping_steps,
        max_excursions,
        sample_limit=sample_limit,
    )
    # Apply overflow patches: these seeds have excursions > int64 that
    # were capped in the numpy array.  Check if any patch beats the
    # current aggregate max.
    for patch in patches:
        if patch.max_excursion > aggregate.max_excursion["value"]:
            aggregate.max_excursion = {"n": patch.seed, "value": patch.max_excursion}
        if patch.total_stopping_time > aggregate.max_total_stopping_time["value"]:
            aggregate.max_total_stopping_time = {"n": patch.seed, "value": patch.total_stopping_time}
        if patch.stopping_time > aggregate.max_stopping_time["value"]:
            aggregate.max_stopping_time = {"n": patch.seed, "value": patch.stopping_time}
    return aggregate


def compute_range_metrics_parallel_odd(
    start: int,
    end: int,
    *,
    start_at: int | None = None,
    sample_limit: int = 12,
    profile: ComputeProfile | None = None,
) -> AggregateMetrics:
    """Process only odd seeds in [start, end] using Numba parallel kernel.

    Mathematically equivalent to verifying the full range: every even n
    maps to n/2 < n, so convergence of all odd seeds implies convergence
    of all integers by strong induction on v2(n).

    Metrics are tracked across odd seeds only.  The ``processed`` count
    reflects actual computational work (odd seeds computed).
    """
    if np is None or njit is None or prange is None:
        raise ValueError("CPU parallel execution requires numpy and numba.")
    if start < 1 or end < start:
        raise ValueError("Invalid run interval.")
    first = start_at if start_at is not None else start
    if first < start or first > end:
        raise ValueError("Checkpoint start is outside the run interval.")

    if set_num_threads is not None:
        total_cores = os.cpu_count() or 1
        effective_percent = _effective_profile_percent(profile, "cpu")
        thread_count = max(1, min(total_cores, round(total_cores * max(effective_percent, 1) / 100)))
        set_num_threads(thread_count)

    # Determine the first odd seed >= first
    first_odd = first if first & 1 else first + 1
    if first_odd > end:
        # No odd seeds in range — return empty aggregate
        return AggregateMetrics(
            processed=0,
            last_processed=end,
            max_total_stopping_time={"n": first, "value": 0},
            max_stopping_time={"n": first, "value": 0},
            max_excursion={"n": first, "value": 0},
            sample_records=[],
        )

    odd_count = ((end - first_odd) // 2) + 1

    total_steps, stopping_steps, max_excursions = _collatz_metrics_parallel_odd(first_odd, odd_count)

    # Overflow fallback: seeds marked -1 by Numba need Python bigint recovery.
    # Cannot use _fallback_overflow_seeds here because seeds are spaced by 2.
    patches: list[_OverflowPatch] = []
    overflow_indices = np.where(total_steps < 0)[0]
    if overflow_indices.size > 0:
        for idx in overflow_indices:
            seed = first_odd + 2 * int(idx)
            m = metrics_accelerated(seed)
            total_steps[idx] = m.total_stopping_time
            stopping_steps[idx] = m.stopping_time
            if m.max_excursion > INT64_MAX:
                max_excursions[idx] = INT64_MAX
                patches.append(_OverflowPatch(seed, m.total_stopping_time, m.stopping_time, m.max_excursion))
            else:
                max_excursions[idx] = m.max_excursion

    # Build aggregate — seeds are first_odd, first_odd+2, ... (not contiguous).
    max_total = {"n": first_odd, "value": -1}
    max_stopping = {"n": first_odd, "value": -1}
    max_excursion_agg = {"n": first_odd, "value": -1}
    sample_records: list[dict] = []

    for idx in range(odd_count):
        seed = first_odd + 2 * idx
        total = int(total_steps[idx])
        stopping = int(stopping_steps[idx])
        excursion = int(max_excursions[idx])

        if total > max_total["value"]:
            max_total = {"n": seed, "value": total}
            if len(sample_records) < sample_limit:
                sample_records.append({"metric": "max_total_stopping_time", **max_total})
        if stopping > max_stopping["value"]:
            max_stopping = {"n": seed, "value": stopping}
            if len(sample_records) < sample_limit:
                sample_records.append({"metric": "max_stopping_time", **max_stopping})
        if excursion > max_excursion_agg["value"]:
            max_excursion_agg = {"n": seed, "value": excursion}
            if len(sample_records) < sample_limit:
                sample_records.append({"metric": "max_excursion", **max_excursion_agg})

    aggregate = AggregateMetrics(
        processed=odd_count,
        last_processed=end,
        max_total_stopping_time=max_total,
        max_stopping_time=max_stopping,
        max_excursion=max_excursion_agg,
        sample_records=sample_records,
    )

    # Apply overflow patches
    for patch in patches:
        if patch.max_excursion > aggregate.max_excursion["value"]:
            aggregate.max_excursion = {"n": patch.seed, "value": patch.max_excursion}
        if patch.total_stopping_time > aggregate.max_total_stopping_time["value"]:
            aggregate.max_total_stopping_time = {"n": patch.seed, "value": patch.total_stopping_time}
        if patch.stopping_time > aggregate.max_stopping_time["value"]:
            aggregate.max_stopping_time = {"n": patch.seed, "value": patch.stopping_time}
    return aggregate


def compute_range_metrics_parallel_descent(
    start: int,
    end: int,
    *,
    start_at: int | None = None,
    sample_limit: int = 12,
    profile: ComputeProfile | None = None,
) -> AggregateMetrics:
    """Full-range standard Collatz descent verifier for validation and GPU fallback."""
    if np is None or njit is None or prange is None:
        raise ValueError("CPU parallel execution requires numpy and numba.")
    if start < 1 or end < start:
        raise ValueError("Invalid run interval.")
    first = start_at if start_at is not None else start
    if first < start or first > end:
        raise ValueError("Checkpoint start is outside the run interval.")

    if set_num_threads is not None:
        total_cores = os.cpu_count() or 1
        effective_percent = _effective_profile_percent(profile, "cpu")
        thread_count = max(1, min(total_cores, round(total_cores * max(effective_percent, 1) / 100)))
        set_num_threads(thread_count)

    size = end - first + 1
    total_steps, stopping_steps, max_excursions = _collatz_metrics_parallel_descent(first, size)
    patches = _fallback_overflow_seeds_descent(first, total_steps, stopping_steps, max_excursions)
    aggregate = _aggregate_metrics_from_arrays(
        first,
        total_steps,
        stopping_steps,
        max_excursions,
        sample_limit=sample_limit,
    )
    for patch in patches:
        if patch.max_excursion > aggregate.max_excursion["value"]:
            aggregate.max_excursion = {"n": patch.seed, "value": patch.max_excursion}
        if patch.total_stopping_time > aggregate.max_total_stopping_time["value"]:
            aggregate.max_total_stopping_time = {"n": patch.seed, "value": patch.total_stopping_time}
        if patch.stopping_time > aggregate.max_stopping_time["value"]:
            aggregate.max_stopping_time = {"n": patch.seed, "value": patch.stopping_time}
    return aggregate


def compute_range_metrics_parallel_descent_odd(
    start: int,
    end: int,
    *,
    start_at: int | None = None,
    sample_limit: int = 12,
    profile: ComputeProfile | None = None,
) -> AggregateMetrics:
    """Odd-only standard Collatz descent verifier for validating cpu-sieve."""
    if np is None or njit is None or prange is None:
        raise ValueError("CPU parallel execution requires numpy and numba.")
    if start < 1 or end < start:
        raise ValueError("Invalid run interval.")
    first = start_at if start_at is not None else start
    if first < start or first > end:
        raise ValueError("Checkpoint start is outside the run interval.")

    if set_num_threads is not None:
        total_cores = os.cpu_count() or 1
        effective_percent = _effective_profile_percent(profile, "cpu")
        thread_count = max(1, min(total_cores, round(total_cores * max(effective_percent, 1) / 100)))
        set_num_threads(thread_count)

    first_odd = first if first & 1 else first + 1
    if first_odd > end:
        return AggregateMetrics(
            processed=0,
            last_processed=end,
            max_total_stopping_time={"n": first, "value": 0},
            max_stopping_time={"n": first, "value": 0},
            max_excursion={"n": first, "value": 0},
            sample_records=[],
        )

    odd_count = ((end - first_odd) // 2) + 1
    total_steps, stopping_steps, max_excursions = _collatz_metrics_parallel_descent_odd(first_odd, odd_count)
    patches = _fallback_overflow_seeds_descent(
        first_odd, total_steps, stopping_steps, max_excursions, stride=2,
    )
    aggregate = _aggregate_metrics_from_strided_arrays(
        first_odd,
        2,
        total_steps,
        stopping_steps,
        max_excursions,
        sample_limit=sample_limit,
    )
    aggregate.last_processed = end
    for patch in patches:
        if patch.max_excursion > aggregate.max_excursion["value"]:
            aggregate.max_excursion = {"n": patch.seed, "value": patch.max_excursion}
        if patch.total_stopping_time > aggregate.max_total_stopping_time["value"]:
            aggregate.max_total_stopping_time = {"n": patch.seed, "value": patch.total_stopping_time}
        if patch.stopping_time > aggregate.max_stopping_time["value"]:
            aggregate.max_stopping_time = {"n": patch.seed, "value": patch.stopping_time}
    return aggregate


def _cpu_sieve_odd_finalize_from_arrays(
    *,
    first_odd: int,
    odd_count: int,
    range_end: int,
    total_steps: np.ndarray,
    stopping_steps: np.ndarray,
    max_excursions: np.ndarray,
    sample_limit: int,
) -> AggregateMetrics:
    """Shared tail: int64 overflow patch + argmax aggregates (Numba and native C paths)."""
    del sample_limit  # same fixed sample cap as historical sieve path
    # Overflow fallback must preserve descent semantics. Using full-orbit
    # metrics here would silently inflate max_total/max_stopping records for
    # large seeds and make validator mismatches look like compute failures.
    patches: list[_OverflowPatch] = []
    overflow_indices = np.where(total_steps < 0)[0]
    if overflow_indices.size > 0:
        for idx in overflow_indices:
            seed = first_odd + 2 * int(idx)
            m = metrics_descent_direct(seed)
            total_steps[idx] = m.total_stopping_time
            stopping_steps[idx] = m.stopping_time
            if m.max_excursion > INT64_MAX:
                max_excursions[idx] = INT64_MAX
                patches.append(_OverflowPatch(seed, m.total_stopping_time, m.stopping_time, m.max_excursion))
            else:
                max_excursions[idx] = m.max_excursion

    # For the sieve kernel, record holders for total vs stopping use the same
    # argmax index as historical Numba output (both tie to total_steps winner).
    best_total_idx = int(np.argmax(total_steps))
    best_total_seed = first_odd + 2 * best_total_idx
    best_total_val = int(total_steps[best_total_idx])
    max_total = {"n": best_total_seed, "value": best_total_val}
    max_stopping = {"n": best_total_seed, "value": best_total_val}

    best_exc_idx = int(np.argmax(max_excursions))
    best_exc_seed = first_odd + 2 * best_exc_idx
    best_exc_val = int(max_excursions[best_exc_idx])
    max_excursion_agg = {"n": best_exc_seed, "value": best_exc_val}

    sample_records: list[dict] = []
    if best_total_val > 0:
        sample_records.append({"metric": "max_total_stopping_time", **max_total})
        sample_records.append({"metric": "max_stopping_time", **max_stopping})
    if best_exc_val > 0:
        sample_records.append({"metric": "max_excursion", **max_excursion_agg})

    aggregate = AggregateMetrics(
        processed=odd_count,
        last_processed=range_end,
        max_total_stopping_time=max_total,
        max_stopping_time=max_stopping,
        max_excursion=max_excursion_agg,
        sample_records=sample_records,
    )

    for patch in patches:
        if patch.max_excursion > aggregate.max_excursion["value"]:
            aggregate.max_excursion = {"n": patch.seed, "value": patch.max_excursion}
        if patch.total_stopping_time > aggregate.max_total_stopping_time["value"]:
            aggregate.max_total_stopping_time = {"n": patch.seed, "value": patch.total_stopping_time}
        if patch.stopping_time > aggregate.max_stopping_time["value"]:
            aggregate.max_stopping_time = {"n": patch.seed, "value": patch.stopping_time}
    return aggregate


def compute_range_metrics_sieve_odd(
    start: int,
    end: int,
    *,
    start_at: int | None = None,
    sample_limit: int = 12,
    profile: ComputeProfile | None = None,
) -> AggregateMetrics:
    """Process only odd seeds with standard Collatz until first descent.

    This is the canonical odd-only descent verifier: every odd seed is checked
    individually, and even seeds are covered by induction on ``v2(n)``.  The
    current implementation is step-by-step; the sieve tables are retained for
    future experimentation but are not part of the correctness contract.

    **Backend:** ``COLLATZ_CPU_SIEVE_BACKEND=auto`` (default), ``numba``, or ``native`` (C shared lib;
    ``native`` raises if the library is missing).
    """
    if cpu_sieve_resolve_backend() == "native":
        return compute_range_metrics_sieve_odd_native(
            start,
            end,
            start_at=start_at,
            sample_limit=sample_limit,
            profile=profile,
        )

    if np is None or njit is None or prange is None:
        raise ValueError("CPU parallel execution requires numpy and numba.")
    if start < 1 or end < start:
        raise ValueError("Invalid run interval.")
    first = start_at if start_at is not None else start
    if first < start or first > end:
        raise ValueError("Checkpoint start is outside the run interval.")

    if set_num_threads is not None:
        total_cores = os.cpu_count() or 1
        effective_percent = _effective_profile_percent(profile, "cpu")
        thread_count = max(1, min(total_cores, round(total_cores * max(effective_percent, 1) / 100)))
        set_num_threads(thread_count)

    first_odd = first if first & 1 else first + 1
    if first_odd > end:
        return AggregateMetrics(
            processed=0,
            last_processed=end,
            max_total_stopping_time={"n": first, "value": 0},
            max_stopping_time={"n": first, "value": 0},
            max_excursion={"n": first, "value": 0},
            sample_records=[],
        )

    odd_count = ((end - first_odd) // 2) + 1
    multipliers, offsets, odd_counts_tbl, safe_limits = _get_sieve_tables()

    total_steps, stopping_steps, max_excursions = _collatz_sieve_parallel_odd(
        first_odd, odd_count, multipliers, offsets, odd_counts_tbl, safe_limits, SIEVE_K
    )

    return _cpu_sieve_odd_finalize_from_arrays(
        first_odd=first_odd,
        odd_count=odd_count,
        range_end=end,
        total_steps=total_steps,
        stopping_steps=stopping_steps,
        max_excursions=max_excursions,
        sample_limit=sample_limit,
    )


def compute_range_metrics_barina_odd(
    start: int,
    end: int,
    *,
    start_at: int | None = None,
    sample_limit: int = 12,
    profile: ComputeProfile | None = None,
) -> AggregateMetrics:
    """EXPERIMENTAL descent-verification using Barina's domain-switching algorithm.

    Uses CTZ + powers-of-3 table instead of if-odd/if-even branching.
    Every odd seed is individually verified — no seeds are skipped.

    IMPORTANT: This kernel is experimental.  The domain-switching
    transformation is mathematically equivalent to standard Collatz for
    convergence, but the step counts and intermediate values differ from
    standard Collatz iteration.  Use cpu-sieve for canonical verification.

    This is a DESCENT verifier, not a full-orbit calculator:
    - total_stopping_time = Barina-domain descent steps (NOT standard Collatz steps)
    - max_excursion = peak in Barina domain (NOT standard Collatz orbit peak)
    """
    if np is None or njit is None or prange is None:
        raise ValueError("CPU parallel execution requires numpy and numba.")
    if start < 1 or end < start:
        raise ValueError("Invalid run interval.")
    first = start_at if start_at is not None else start
    if first < start or first > end:
        raise ValueError("Checkpoint start is outside the run interval.")

    if set_num_threads is not None:
        total_cores = os.cpu_count() or 1
        effective_percent = _effective_profile_percent(profile, "cpu")
        thread_count = max(1, min(total_cores, round(total_cores * max(effective_percent, 1) / 100)))
        set_num_threads(thread_count)

    first_odd = first if first & 1 else first + 1
    if first_odd > end:
        return AggregateMetrics(
            processed=0,
            last_processed=end,
            max_total_stopping_time={"n": first, "value": 0},
            max_stopping_time={"n": first, "value": 0},
            max_excursion={"n": first, "value": 0},
            sample_records=[],
        )

    odd_count = ((end - first_odd) // 2) + 1
    pow3 = _get_pow3_table()

    stopping_steps, max_excursions = _collatz_barina_parallel_odd(
        first_odd, odd_count, pow3
    )

    # Overflow fallback
    overflow_indices = np.where(stopping_steps < 0)[0]
    if overflow_indices.size > 0:
        for idx in overflow_indices:
            seed = first_odd + 2 * int(idx)
            m = metrics_accelerated(seed)
            stopping_steps[idx] = m.stopping_time
            if m.max_excursion > INT64_MAX:
                max_excursions[idx] = INT64_MAX
            else:
                max_excursions[idx] = m.max_excursion

    # Fast NumPy aggregation
    best_stop_idx = int(np.argmax(stopping_steps))
    best_stop_seed = first_odd + 2 * best_stop_idx
    best_stop_val = int(stopping_steps[best_stop_idx])
    max_stopping = {"n": best_stop_seed, "value": best_stop_val}

    best_exc_idx = int(np.argmax(max_excursions))
    best_exc_seed = first_odd + 2 * best_exc_idx
    best_exc_val = int(max_excursions[best_exc_idx])
    max_excursion_agg = {"n": best_exc_seed, "value": best_exc_val}

    sample_records: list[dict] = []
    if best_stop_val > 0:
        sample_records.append({"metric": "max_total_stopping_time", **max_stopping})
        sample_records.append({"metric": "max_stopping_time", **max_stopping})
    if best_exc_val > 0:
        sample_records.append({"metric": "max_excursion", **max_excursion_agg})

    return AggregateMetrics(
        processed=odd_count,
        last_processed=end,
        max_total_stopping_time=max_stopping,
        max_stopping_time=max_stopping,
        max_excursion=max_excursion_agg,
        sample_records=sample_records,
    )


def _compute_range_metrics_gpu_sieve_mps(
    start: int,
    end: int,
    *,
    start_at: int | None = None,
    sample_limit: int = 12,
    profile: ComputeProfile | None = None,
) -> AggregateMetrics:
    """Apple MPS odd-only sieve — same aggregates as cpu-sieve / CUDA gpu-sieve.

    Uses **streaming reduction** over MPS chunks: O(chunk) host memory only.
    The previous implementation materialised full ``odd_count`` int32/int64 arrays
    per checkpoint (hundreds of millions of seeds → multi-GB RAM + slow copies),
    which made long ``gpu-sieve`` runs look "stuck" at sub-1M/s on Apple Silicon.
    """
    from . import mps_collatz

    import torch

    if np is None:
        raise ValueError("numpy is required for MPS GPU execution.")
    if not mps_collatz.mps_accelerated_available():
        raise ValueError("MPS is not available.")
    first = start_at if start_at is not None else start
    if first < start or first > end:
        raise ValueError("Checkpoint start is outside the run interval.")

    first_odd = first if first & 1 else first + 1
    if first_odd > end:
        return AggregateMetrics(
            processed=0,
            last_processed=end,
            max_total_stopping_time={"n": first, "value": 0},
            max_stopping_time={"n": first, "value": 0},
            max_excursion={"n": first, "value": 0},
            sample_records=[],
        )

    odd_count = ((end - first_odd) // 2) + 1
    chunk_size = _mps_sieve_chunk_size()
    device = torch.device("mps")
    fo = torch.tensor(int(first_odd), dtype=torch.int64, device=device)

    best_total_val = -1
    best_total_seed = first_odd
    best_exc_val = -1
    best_exc_seed = first_odd
    patches: list[_OverflowPatch] = []

    offset = 0
    while offset < odd_count:
        chunk = min(chunk_size, odd_count - offset)
        idx = torch.arange(offset, offset + chunk, dtype=torch.int64, device=device)
        originals = fo + 2 * idx
        total, _stopping, max_exc = mps_collatz._mps_batch_sieve_descent(originals)
        total_np = total.cpu().numpy()
        max_exc_np = max_exc.cpu().numpy()

        overflow_indices = np.where(total_np < 0)[0]
        if overflow_indices.size > 0:
            for j in overflow_indices:
                j = int(j)
                seed = first_odd + 2 * (offset + j)
                m = metrics_descent_direct(seed)
                total_np[j] = m.total_stopping_time
                if m.max_excursion > INT64_MAX:
                    max_exc_np[j] = INT64_MAX
                    patches.append(_OverflowPatch(seed, m.total_stopping_time, m.stopping_time, m.max_excursion))
                else:
                    max_exc_np[j] = m.max_excursion

        lt = int(np.argmax(total_np))
        tv = int(total_np[lt])
        if tv > best_total_val:
            best_total_val = tv
            best_total_seed = first_odd + 2 * (offset + lt)

        le = int(np.argmax(max_exc_np))
        ev = int(max_exc_np[le])
        if ev > best_exc_val:
            best_exc_val = ev
            best_exc_seed = first_odd + 2 * (offset + le)

        offset += chunk

    max_total = {"n": best_total_seed, "value": best_total_val}
    max_stopping = {"n": best_total_seed, "value": best_total_val}

    max_excursion_agg = {"n": best_exc_seed, "value": best_exc_val}

    sample_records: list[dict] = []
    if best_total_val > 0:
        sample_records.append({"metric": "max_total_stopping_time", **max_total})
        sample_records.append({"metric": "max_stopping_time", **max_stopping})
    if best_exc_val > 0:
        sample_records.append({"metric": "max_excursion", **max_excursion_agg})

    aggregate = AggregateMetrics(
        processed=odd_count,
        last_processed=end,
        max_total_stopping_time=max_total,
        max_stopping_time=max_stopping,
        max_excursion=max_excursion_agg,
        sample_records=sample_records,
    )

    for patch in patches:
        if patch.max_excursion > aggregate.max_excursion["value"]:
            aggregate.max_excursion = {"n": patch.seed, "value": patch.max_excursion}
        if patch.total_stopping_time > aggregate.max_total_stopping_time["value"]:
            aggregate.max_total_stopping_time = {"n": patch.seed, "value": patch.total_stopping_time}
        if patch.stopping_time > aggregate.max_stopping_time["value"]:
            aggregate.max_stopping_time = {"n": patch.seed, "value": patch.stopping_time}
    return aggregate


def compute_range_metrics_gpu_sieve(
    start: int,
    end: int,
    *,
    start_at: int | None = None,
    sample_limit: int = 12,
    profile: ComputeProfile | None = None,
) -> AggregateMetrics:
    """GPU odd-only standard Collatz descent verifier.

    Processes only odd seeds (like cpu-sieve) for 2x throughput improvement.
    Even seeds are covered by induction on v2(n).
    """
    if np is None:
        raise ValueError("numpy is required for GPU sieve execution.")
    metal_only_gate = False
    if platform.system() == "Darwin":
        from .gpu_sieve_metal_runtime import (
            gpu_sieve_metal_without_torch_allowed,
            native_metal_sieve_available,
        )

        metal_only_gate = bool(
            gpu_sieve_metal_without_torch_allowed() and native_metal_sieve_available()
        )
    if not gpu_execution_ready() and not metal_only_gate:
        raise ValueError("GPU execution is not available on this machine.")
    silence_numba_cuda_info()
    if start < 1 or end < start:
        raise ValueError("Invalid run interval.")
    first = start_at if start_at is not None else start
    if first < start or first > end:
        raise ValueError("Checkpoint start is outside the run interval.")

    first_odd = first if first & 1 else first + 1
    odd_count = ((end - first_odd) // 2) + 1 if first_odd <= end else 0
    if odd_count <= 0:
        return AggregateMetrics(
            processed=0,
            last_processed=end,
            max_total_stopping_time={"n": first, "value": 0},
            max_stopping_time={"n": first, "value": 0},
            max_excursion={"n": first, "value": 0},
            sample_records=[],
        )

    if cuda is not None and cuda_gpu_execution_ready():
        size = odd_count
        threads_per_block = _gpu_threads_per_block(profile)
        seeds_per_thread = GPU_SEEDS_PER_THREAD
        total_threads = math.ceil(size / seeds_per_thread)
        blocks_per_grid = math.ceil(total_threads / threads_per_block)

        multipliers, offsets, odd_counts_tbl, safe_limits = _get_sieve_tables()
        d_mul = cuda.to_device(multipliers)
        d_off = cuda.to_device(offsets)
        d_odc = cuda.to_device(odd_counts_tbl)
        d_safe = cuda.to_device(safe_limits)

        d_block_total_values = cuda.device_array(blocks_per_grid, dtype=np.int32)
        d_block_total_ns = cuda.device_array(blocks_per_grid, dtype=np.int64)
        d_block_stopping_values = cuda.device_array(blocks_per_grid, dtype=np.int32)
        d_block_stopping_ns = cuda.device_array(blocks_per_grid, dtype=np.int64)
        d_block_excursion_values = cuda.device_array(blocks_per_grid, dtype=np.int64)
        d_block_excursion_ns = cuda.device_array(blocks_per_grid, dtype=np.int64)
        d_block_overflow_ns = cuda.device_array(blocks_per_grid, dtype=np.int64)

        _collatz_sieve_gpu_kernel[blocks_per_grid, threads_per_block](
            first_odd,
            size,
            seeds_per_thread,
            SIEVE_K,
            d_mul, d_off, d_odc, d_safe,
            d_block_total_values,
            d_block_total_ns,
            d_block_stopping_values,
            d_block_stopping_ns,
            d_block_excursion_values,
            d_block_excursion_ns,
            d_block_overflow_ns,
        )

        block_total_values = d_block_total_values.copy_to_host()
        block_total_ns = d_block_total_ns.copy_to_host()
        block_stopping_values = d_block_stopping_values.copy_to_host()
        block_stopping_ns = d_block_stopping_ns.copy_to_host()
        block_excursion_values = d_block_excursion_values.copy_to_host()
        block_excursion_ns = d_block_excursion_ns.copy_to_host()
        block_overflow_ns = d_block_overflow_ns.copy_to_host()

        first_overflow = int(block_overflow_ns[block_overflow_ns > 0].min()) if (block_overflow_ns > 0).any() else 0

        if first_overflow > 0:
            return compute_range_metrics_parallel_descent(
                start, end, start_at=start_at, sample_limit=sample_limit, profile=profile
            )

        agg = _aggregate_metrics_from_block_summaries(
            first_odd,
            odd_count,
            block_total_values,
            block_total_ns,
            block_stopping_values,
            block_stopping_ns,
            block_excursion_values,
            block_excursion_ns,
            sample_limit=sample_limit,
        )
        agg.processed = odd_count
        agg.last_processed = end
        return agg

    from . import mps_collatz
    from .gpu_sieve_metal_runtime import (
        compute_range_metrics_gpu_sieve_metal,
        gpu_sieve_backend_mode,
        gpu_sieve_metal_without_torch_allowed,
        native_metal_sieve_available,
        should_use_native_metal_sieve,
    )

    def _try_native_metal_sieve_path() -> AggregateMetrics | None:
        mode = gpu_sieve_backend_mode()
        use_metal = should_use_native_metal_sieve()
        if not use_metal:
            return None
        try:
            return compute_range_metrics_gpu_sieve_metal(
                start,
                end,
                start_at=start_at,
                sample_limit=sample_limit,
                profile=profile,
            )
        except Exception as exc:
            if mode == "metal":
                raise
            logger.warning(
                "Native Metal gpu-sieve failed; falling back to PyTorch MPS: %s",
                exc,
            )
        return None

    if mps_collatz.mps_accelerated_available():
        metal_agg = _try_native_metal_sieve_path()
        if metal_agg is not None:
            return metal_agg
        return _compute_range_metrics_gpu_sieve_mps(
            start,
            end,
            start_at=start_at,
            sample_limit=sample_limit,
            profile=profile,
        )

    if (
        platform.system() == "Darwin"
        and gpu_sieve_metal_without_torch_allowed()
        and native_metal_sieve_available()
    ):
        metal_agg = _try_native_metal_sieve_path()
        if metal_agg is not None:
            return metal_agg

    raise ValueError("GPU execution is not available on this machine.")


def compute_range_metrics(
    start: int,
    end: int,
    *,
    kernel: str = "cpu-direct",
    start_at: int | None = None,
    sample_limit: int = 12,
    profile: ComputeProfile | None = None,
) -> AggregateMetrics:
    if start < 1 or end < start:
        raise ValueError("Invalid run interval.")
    first = start_at if start_at is not None else start
    if first < start or first > end:
        raise ValueError("Checkpoint start is outside the run interval.")
    if kernel == CPU_PARALLEL_KERNEL:
        return compute_range_metrics_parallel(
            start,
            end,
            start_at=start_at,
            sample_limit=sample_limit,
            profile=profile,
        )
    if kernel == CPU_PARALLEL_ODD_KERNEL:
        return compute_range_metrics_parallel_odd(
            start,
            end,
            start_at=start_at,
            sample_limit=sample_limit,
            profile=profile,
        )
    if kernel == CPU_SIEVE_KERNEL:
        return compute_range_metrics_sieve_odd(
            start,
            end,
            start_at=start_at,
            sample_limit=sample_limit,
            profile=profile,
        )
    if kernel == CPU_BARINA_KERNEL:
        return compute_range_metrics_barina_odd(
            start,
            end,
            start_at=start_at,
            sample_limit=sample_limit,
            profile=profile,
        )
    if kernel == GPU_KERNEL:
        return compute_range_metrics_gpu(
            start,
            end,
            start_at=start_at,
            sample_limit=sample_limit,
            profile=profile,
        )
    if kernel == GPU_SIEVE_KERNEL:
        return compute_range_metrics_gpu_sieve(
            start,
            end,
            start_at=start_at,
            sample_limit=sample_limit,
            profile=profile,
        )
    if kernel not in {CPU_DIRECT_KERNEL, CPU_ACCELERATED_KERNEL}:
        raise ValueError(f"Unsupported execution kernel: {kernel}")

    max_total = {"n": first, "value": -1}
    max_stopping = {"n": first, "value": -1}
    max_excursion = {"n": first, "value": -1}
    sample_records: list[dict] = []
    processed = 0
    last_processed = first - 1
    metric_fn = metrics_direct if kernel == CPU_DIRECT_KERNEL else metrics_accelerated

    for value in range(first, end + 1):
        metrics = metric_fn(value)
        processed += 1
        last_processed = value

        if metrics.total_stopping_time > max_total["value"]:
            max_total = {"n": value, "value": metrics.total_stopping_time}
            if len(sample_records) < sample_limit:
                sample_records.append({"metric": "max_total_stopping_time", **max_total})

        if metrics.stopping_time > max_stopping["value"]:
            max_stopping = {"n": value, "value": metrics.stopping_time}
            if len(sample_records) < sample_limit:
                sample_records.append({"metric": "max_stopping_time", **max_stopping})

        if metrics.max_excursion > max_excursion["value"]:
            max_excursion = {"n": value, "value": metrics.max_excursion}
            if len(sample_records) < sample_limit:
                sample_records.append({"metric": "max_excursion", **max_excursion})

    return AggregateMetrics(
        processed=processed,
        last_processed=last_processed,
        max_total_stopping_time=max_total,
        max_stopping_time=max_stopping,
        max_excursion=max_excursion,
        sample_records=sample_records,
    )

