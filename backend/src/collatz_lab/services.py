from __future__ import annotations

import json
import logging
import math
import os
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
from .repository import LabRepository, sha256_text, utc_now
from .schemas import ArtifactKind, ComputeProfile, ModularProbeResult, Run, RunStatus

logger = logging.getLogger("collatz_lab.services")

class _CheckpointWriter:
    """Async DB checkpoint writer for execute_run.

    Runs ``repository.update_run`` in a single background thread so the
    GPU (or CPU kernel) can start the next batch while SQLite commits
    the previous one.

    At most one write is in-flight at a time.  ``drain()`` blocks until
    the pending write completes.  This guarantees crash-safety: if the
    process dies mid-batch, the *previous* checkpoint is already committed.
    """

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="checkpoint")
        self._pending: Future | None = None

    def drain(self) -> None:
        """Block until the in-flight write finishes.  Re-raises on failure."""
        if self._pending is not None:
            self._pending.result()
            self._pending = None

    def submit(self, repository: LabRepository, run_id: str, **kwargs) -> None:
        """Drain the previous write, then submit a new one."""
        self.drain()
        self._pending = self._executor.submit(repository.update_run, run_id, **kwargs)

    def shutdown(self) -> None:
        """Drain pending work and shut down the thread pool."""
        self.drain()
        self._executor.shutdown(wait=True)


INT64_MAX = (1 << 63) - 1
COLLATZ_INT64_ODD_STEP_LIMIT = (INT64_MAX - 1) // 3
OVERFLOW_RECOVERY_OWNER = "overflow-recovery"
LEGACY_VALIDATION_RERUN_OWNER = "legacy-revalidation"
LEGACY_VALIDATION_RERUN_PREFIX = "legacy-rerun-"
# Safety bound for GPU/parallel kernels: the longest known Collatz orbit
# for seeds below 2^64 converges well within 2500 steps.  This generous
# limit prevents GPU thread hangs if a bug or counterexample is found.
MAX_KERNEL_STEPS = 100_000


def collatz_step(value: int) -> int:
    if value < 1:
        raise ValueError("Collatz is defined only for positive integers.")
    return value // 2 if value % 2 == 0 else (3 * value) + 1


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


@dataclass
class NumberMetrics:
    total_stopping_time: int
    stopping_time: int
    max_excursion: int


@dataclass
class AggregateMetrics:
    processed: int
    last_processed: int
    max_total_stopping_time: dict
    max_stopping_time: dict
    max_excursion: dict
    sample_records: list[dict]


def metrics_direct(value: int) -> NumberMetrics:
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


def metrics_descent_direct(value: int) -> NumberMetrics:
    """Compute standard Collatz metrics only until the orbit first drops below the seed.

    This is the exact semantic used by the sieve-style verification kernels:
    convergence is proven once the orbit reaches any smaller value, because that
    value is already covered by ascending-order verification.
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


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _effective_profile_percent(profile: ComputeProfile | None, hardware: str) -> int:
    system_percent = 100 if profile is None else max(0, min(100, int(profile.system_percent)))
    hardware_percent = 100
    if profile is not None:
        hardware_percent = max(
            0,
            min(100, int(profile.cpu_percent if hardware == "cpu" else profile.gpu_percent)),
        )
    return max(0, min(100, round((system_percent * hardware_percent) / 100)))


def _compute_budget_throttle_seconds(
    *,
    hardware: str,
    profile: ComputeProfile | None,
    compute_sec: float,
) -> float:
    """Idle time so average duty cycle matches the compute profile.

    Batch size and CPU thread count both scale with the same effective percent,
    so wall time per batch stays ~flat between 50% and 100% without extra
    idle. GPU batches also keep the device saturated for the whole kernel.
    Sleeping after each batch restores a visible slowdown when the slider is
    below 100%.
    """
    if compute_sec <= 0:
        return 0.0
    lane = "gpu" if hardware == "gpu" else "cpu"
    effective = _effective_profile_percent(profile, lane)
    if effective >= 100:
        return 0.0
    eff = max(1, effective)
    return compute_sec * (100.0 / eff - 1.0)


def _gpu_threads_per_block(profile: ComputeProfile | None = None) -> int:
    effective_percent = _effective_profile_percent(profile, "gpu")
    default_value = 128 if effective_percent <= 25 else 256
    value = _positive_int_env("COLLATZ_GPU_THREADS_PER_BLOCK", default_value)
    # Capped at 256 — shared memory arrays in the kernel are sized 256.
    # With 256 threads/block: ~11KB shared mem → 4 blocks/SM on RTX 4060 Ti
    # (vs 1 block/SM at 1024), giving much better latency hiding and occupancy.
    return max(64, min(256, value))


def compute_range_metrics_gpu(
    start: int,
    end: int,
    *,
    start_at: int | None = None,
    sample_limit: int = 12,
    profile: ComputeProfile | None = None,
) -> AggregateMetrics:
    if not gpu_execution_ready() or cuda is None or np is None:
        raise ValueError("GPU execution is not available on this machine.")
    silence_numba_cuda_info()
    if start < 1 or end < start:
        raise ValueError("Invalid run interval.")
    first = start_at if start_at is not None else start
    if first < start or first > end:
        raise ValueError("Checkpoint start is outside the run interval.")

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
        # GPU detected overflow seeds — fall back to cpu-parallel with
        # Python bigint recovery for the entire batch so we get per-seed
        # accuracy instead of only per-block summaries.
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
    multipliers, offsets, odd_counts_tbl, safe_limits = _get_sieve_tables()

    total_steps, stopping_steps, max_excursions = _collatz_sieve_parallel_odd(
        first_odd, odd_count, multipliers, offsets, odd_counts_tbl, safe_limits, SIEVE_K
    )

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

    # Fast NumPy aggregation — replaces O(N) Python loop with vectorised ops.
    # For the sieve kernel total_steps == stopping_steps (early termination).
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
    if not gpu_execution_ready() or cuda is None or np is None:
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
    # Fix processed count and last_processed to reflect the full range
    # (even seeds are covered by induction, same as cpu-sieve).
    agg.processed = odd_count
    agg.last_processed = end
    return agg


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


def _effective_checkpoint_interval(
    kernel: str,
    checkpoint_interval: int,
    profile: ComputeProfile | None = None,
) -> int:
    cpu_percent = _effective_profile_percent(profile, "cpu")
    gpu_percent = _effective_profile_percent(profile, "gpu")
    minimums = {
        CPU_ACCELERATED_KERNEL: max(
            50_000,
            round(_positive_int_env("COLLATZ_CPU_ACCELERATED_BATCH_SIZE", 250_000) * max(cpu_percent, 5) / 100),
        ),
        CPU_PARALLEL_KERNEL: max(
            1_000_000,
            round(_positive_int_env("COLLATZ_CPU_PARALLEL_BATCH_SIZE", 50_000_000) * max(cpu_percent, 5) / 100),
        ),
        CPU_PARALLEL_ODD_KERNEL: max(
            2_000_000,
            round(_positive_int_env("COLLATZ_CPU_PARALLEL_ODD_BATCH_SIZE", 100_000_000) * max(cpu_percent, 5) / 100),
        ),
        GPU_KERNEL: max(
            10_000_000,
            round(_positive_int_env("COLLATZ_GPU_BATCH_SIZE", 100_000_000) * max(gpu_percent, 5) / 100),
        ),
        CPU_SIEVE_KERNEL: max(
            5_000_000,
            round(_positive_int_env("COLLATZ_CPU_SIEVE_BATCH_SIZE", 200_000_000) * max(cpu_percent, 5) / 100),
        ),
        CPU_BARINA_KERNEL: max(
            5_000_000,
            round(_positive_int_env("COLLATZ_CPU_BARINA_BATCH_SIZE", 200_000_000) * max(cpu_percent, 5) / 100),
        ),
        GPU_SIEVE_KERNEL: max(
            25_000_000,
            round(_positive_int_env("COLLATZ_GPU_SIEVE_BATCH_SIZE", 500_000_000) * max(gpu_percent, 5) / 100),
        ),
    }
    return max(checkpoint_interval, minimums.get(kernel, checkpoint_interval))


def _is_overflow_guard_failure(run: Run) -> bool:
    return (
        run.direction_slug == "verification"
        and run.status == RunStatus.FAILED
        and "overflow guard triggered" in (run.summary or "").lower()
    )


def _prune_duplicate_overflow_recovery_runs(repository: LabRepository) -> list[str]:
    queued_duplicates: dict[tuple[str, int, int, str, str, str], list[Run]] = {}
    for run in repository.list_runs():
        if run.owner != OVERFLOW_RECOVERY_OWNER or not run.name.startswith("recover-"):
            continue
        key = (run.name, run.range_start, run.range_end, run.hardware, run.kernel, run.owner)
        queued_duplicates.setdefault(key, []).append(run)

    removed_ids: list[str] = []
    for duplicates in queued_duplicates.values():
        if len(duplicates) < 2:
            continue
        duplicates = sorted(duplicates, key=lambda item: (item.created_at, item.id))
        keep = duplicates[0]
        for duplicate in duplicates[1:]:
            # Only delete synthetic duplicates that never started; preserve real work.
            if duplicate.status == RunStatus.QUEUED:
                repository.delete_run(duplicate.id)
                removed_ids.append(duplicate.id)
            elif keep.status == RunStatus.QUEUED and duplicate.status in {
                RunStatus.RUNNING,
                RunStatus.COMPLETED,
                RunStatus.VALIDATED,
            }:
                repository.delete_run(keep.id)
                removed_ids.append(keep.id)
                keep = duplicate
    return removed_ids


def _ensure_overflow_recovery_runs(
    repository: LabRepository,
    *,
    run_ids: set[str] | None = None,
) -> list[str]:
    _prune_duplicate_overflow_recovery_runs(repository)
    runs = repository.list_runs()
    runs_by_name = {run.name: run for run in runs}
    created_ids: list[str] = []

    for failed_run in sorted(runs, key=lambda item: (item.created_at, item.id)):
        if not _is_overflow_guard_failure(failed_run):
            continue
        if run_ids is not None and failed_run.id not in run_ids:
            continue

        current = repository.get_run(failed_run.id)
        checkpoint = current.checkpoint or {}
        metrics = current.metrics or {}
        last_processed = int(
            checkpoint.get("last_processed")
            or metrics.get("last_processed")
            or (current.range_start - 1)
        )

        if last_processed >= current.range_start:
            prefix_name = f"recover-prefix-{current.id}"
            if prefix_name not in runs_by_name:
                prefix_run = repository.create_run(
                    direction_slug=current.direction_slug,
                    name=prefix_name,
                    range_start=current.range_start,
                    range_end=last_processed,
                    kernel=current.kernel,
                    owner=OVERFLOW_RECOVERY_OWNER,
                    hardware=current.hardware,
                )
                checksum = (
                    sha256_text(json.dumps(metrics, sort_keys=True))
                    if metrics
                    else prefix_run.checksum
                )
                prefix_run = repository.update_run(
                    prefix_run.id,
                    status=RunStatus.COMPLETED,
                    checkpoint={
                        "next_value": last_processed + 1,
                        "last_processed": last_processed,
                    },
                    metrics=metrics,
                    summary=(
                        f"Recovered exact prefix from {current.id} after overflow guard "
                        f"stopped the original run. Covers {current.range_start}-{last_processed}."
                    ),
                    checksum=checksum,
                    started_at=current.started_at or prefix_run.created_at,
                    finished_at=utc_now(),
                )
                runs_by_name[prefix_name] = prefix_run
                created_ids.append(prefix_run.id)

        recovery_start = max(current.range_start, last_processed + 1)
        if recovery_start <= current.range_end:
            tail_name = f"recover-tail-{current.id}"
            if tail_name not in runs_by_name:
                recovery_run = repository.create_run(
                    direction_slug=current.direction_slug,
                    name=tail_name,
                    range_start=recovery_start,
                    range_end=current.range_end,
                    kernel=CPU_PARALLEL_KERNEL,
                    owner=OVERFLOW_RECOVERY_OWNER,
                    hardware="cpu",
                )
                recovery_run = repository.update_run(
                    recovery_run.id,
                    summary=(
                        f"Exact CPU recovery queued for the uncovered tail after {current.id} "
                        f"hit the signed-64-bit overflow frontier on {current.hardware}. "
                        f"Covers {recovery_start}-{current.range_end}."
                    ),
                )
                runs_by_name[tail_name] = recovery_run
                created_ids.append(recovery_run.id)

    return created_ids


def queue_continuous_verification_runs(
    repository: LabRepository,
    *,
    supported_hardware: list[str] | None = None,
    owner: str = "gemini-autopilot",
) -> list[str]:
    _ensure_overflow_recovery_runs(repository)
    runs = repository.list_runs()
    allowed_hardware = set(supported_hardware or ["cpu", "gpu"])
    overflow_history_hardware = {
        run.hardware
        for run in runs
        if _is_overflow_guard_failure(run)
    }
    profile = repository.get_compute_profile()
    if not profile.continuous_enabled:
        return []
    active_runs = [
        run
        for run in runs
        if run.status.value in {"queued", "running"}
    ]
    active_cpu = any(run.hardware == "cpu" for run in active_runs)
    active_gpu = any(run.hardware == "gpu" for run in active_runs)
    covered_runs = [
        run
        for run in runs
        if run.status in {RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.COMPLETED, RunStatus.VALIDATED}
    ]
    next_start = max((run.range_end for run in covered_runs), default=0) + 1
    queued_ids: list[str] = []

    cpu_intensity = _effective_profile_percent(profile, "cpu")
    gpu_intensity = _effective_profile_percent(profile, "gpu")
    # Dynamic span: look at last 3 completed CPU sieve runs and scale up if fast
    recent_cpu_runs = [
        r for r in runs
        if r.hardware == "cpu"
        and r.kernel == CPU_SIEVE_KERNEL
        and r.status in {RunStatus.COMPLETED, RunStatus.VALIDATED}
        and r.started_at
        and r.finished_at
    ][-3:]

    base_cpu_span = 2_000_000_000
    if len(recent_cpu_runs) >= 2:
        durations = []
        for r in recent_cpu_runs:
            try:
                start = datetime.fromisoformat(r.started_at.replace("Z", "+00:00"))
                finish = datetime.fromisoformat(r.finished_at.replace("Z", "+00:00"))
                durations.append((finish - start).total_seconds())
            except Exception:
                pass
        if durations:
            avg_secs = sum(durations) / len(durations)
            run_size = recent_cpu_runs[-1].range_end - recent_cpu_runs[-1].range_start + 1
            if avg_secs > 0 and run_size > 0:
                # Target: runs should take ~120 seconds for good checkpoint cadence
                # No upper cap — span grows freely until runs naturally reach ~120s
                target_span = int(run_size * 120 / max(avg_secs, 5))
                base_cpu_span = max(500_000_000, target_span)

    # Sieve kernel uses standard Collatz steps with early termination.
    # Every odd seed is individually verified — no seeds are skipped.
    cpu_span = max(100_000_000, round(base_cpu_span * max(cpu_intensity, 5) / 100))

    # Dynamic GPU span: same strategy as CPU — target ~120s per run
    # GPU sieve is full-range Numba-bound, so it's slower than CPU per seed.
    recent_gpu_runs = [
        r for r in runs
        if r.hardware == "gpu"
        and r.kernel == GPU_SIEVE_KERNEL
        and r.status in {RunStatus.COMPLETED, RunStatus.VALIDATED}
        and r.started_at
        and r.finished_at
    ][-3:]
    base_gpu_span = 1_000_000_000
    if len(recent_gpu_runs) >= 2:
        gpu_durations = []
        for r in recent_gpu_runs:
            try:
                start = datetime.fromisoformat(r.started_at.replace("Z", "+00:00"))
                finish = datetime.fromisoformat(r.finished_at.replace("Z", "+00:00"))
                gpu_durations.append((finish - start).total_seconds())
            except Exception:
                pass
        if gpu_durations:
            avg_gpu_secs = sum(gpu_durations) / len(gpu_durations)
            gpu_run_size = recent_gpu_runs[-1].range_end - recent_gpu_runs[-1].range_start + 1
            if avg_gpu_secs > 0 and gpu_run_size > 0:
                # No upper cap — span grows freely until runs naturally reach ~120s
                target_gpu_span = int(gpu_run_size * 120 / max(avg_gpu_secs, 10))
                base_gpu_span = max(250_000_000, target_gpu_span)
    gpu_span = max(250_000_000, round(base_gpu_span * max(gpu_intensity, 5) / 100))

    if "cpu" in allowed_hardware and cpu_intensity > 0 and not active_cpu:
        cpu_end = next_start + cpu_span - 1
        run = repository.create_run(
            direction_slug="verification",
            name="autopilot-continuous-cpu",
            range_start=next_start,
            range_end=cpu_end,
            kernel=CPU_SIEVE_KERNEL,
            hardware="cpu",
            owner=owner,
        )
        queued_ids.append(run.id)
        next_start = cpu_end + 1

    if "gpu" in allowed_hardware and gpu_intensity > 0 and not active_gpu:
        gpu_end = next_start + gpu_span - 1
        run = repository.create_run(
            direction_slug="verification",
            name="autopilot-continuous-gpu",
            range_start=next_start,
            range_end=gpu_end,
            kernel=GPU_SIEVE_KERNEL,
            hardware="gpu",
            owner=owner,
        )
        queued_ids.append(run.id)

    return queued_ids


def _is_legacy_validation_failure(run: Run) -> bool:
    return (
        run.direction_slug == "verification"
        and run.status == RunStatus.FAILED
        and run.owner != LEGACY_VALIDATION_RERUN_OWNER
        and str(run.summary or "").startswith("Validation failed:")
    )


def queue_legacy_validation_reruns(
    repository: LabRepository,
    *,
    supported_hardware: list[str] | None = None,
    limit: int = 4,
) -> list[str]:
    allowed_hardware = set(supported_hardware or ["cpu", "gpu"])
    runs = repository.list_runs()
    existing_names = {run.name for run in runs}
    existing_exact_ranges = {
        (run.range_start, run.range_end, run.kernel, run.hardware)
        for run in runs
        if run.status in {RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.COMPLETED, RunStatus.VALIDATED}
    }
    queued_ids: list[str] = []

    candidates = sorted(
        [
            run
            for run in runs
            if _is_legacy_validation_failure(run)
            and run.hardware in allowed_hardware
            and f"{LEGACY_VALIDATION_RERUN_PREFIX}{run.id}" not in existing_names
            and (run.range_start, run.range_end, run.kernel, run.hardware) not in existing_exact_ranges
        ],
        key=lambda run: (run.range_start, run.created_at or "", run.id),
    )

    for source in candidates[: max(0, limit)]:
        rerun = repository.create_run(
            direction_slug=source.direction_slug,
            name=f"{LEGACY_VALIDATION_RERUN_PREFIX}{source.id}",
            range_start=source.range_start,
            range_end=source.range_end,
            kernel=source.kernel,
            owner=LEGACY_VALIDATION_RERUN_OWNER,
            code_version=source.code_version,
            hardware=source.hardware,
        )
        queued_ids.append(rerun.id)
        existing_names.add(rerun.name)
        existing_exact_ranges.add((source.range_start, source.range_end, source.kernel, source.hardware))
        if "Legacy revalidation queued via" not in (source.summary or ""):
            repository.update_run(
                source.id,
                summary=(
                    f"{source.summary.rstrip()} "
                    f"Legacy revalidation queued via {rerun.id}."
                ).strip(),
            )

    return queued_ids


def annotate_legacy_validation_failures(
    repository: LabRepository,
    *,
    limit: int = 64,
) -> int:
    runs = repository.list_runs()
    successors_by_signature: dict[tuple[int, int, str, str], list[Run]] = {}
    for run in runs:
        if run.status not in {RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.COMPLETED, RunStatus.VALIDATED}:
            continue
        signature = (run.range_start, run.range_end, run.kernel, run.hardware)
        successors_by_signature.setdefault(signature, []).append(run)

    updated = 0
    for failed in runs:
        if not _is_legacy_validation_failure(failed):
            continue
        if "Superseded by" in (failed.summary or ""):
            continue
        signature = (failed.range_start, failed.range_end, failed.kernel, failed.hardware)
        successors = [
            run
            for run in successors_by_signature.get(signature, [])
            if run.id != failed.id
        ]
        if not successors:
            continue
        successors.sort(
            key=lambda run: (
                0 if run.status == RunStatus.VALIDATED else 1,
                0 if run.status == RunStatus.COMPLETED else 1,
                run.created_at or "",
                run.id,
            )
        )
        successor = successors[0]
        repository.update_run(
            failed.id,
            summary=(
                f"{failed.summary.rstrip()} "
                f"Superseded by {successor.id} ({successor.status.value}) on the current code path."
            ).strip(),
        )
        updated += 1
        if updated >= max(0, limit):
            break

    return updated


def _aggregate_validation_payload(metrics: AggregateMetrics) -> dict:
    return {
        "processed": metrics.processed,
        "last_processed": metrics.last_processed,
        "max_total_stopping_time": metrics.max_total_stopping_time,
        "max_stopping_time": metrics.max_stopping_time,
        "max_excursion": metrics.max_excursion,
    }


def _run_completion_summary(run: Run, aggregate: dict) -> str:
    if run.kernel == CPU_SIEVE_KERNEL:
        return (
            f"Completed odd-seed descent verification on {run.hardware} for range "
            f"{run.range_start}-{run.range_end}; max descent steps at "
            f"{aggregate['max_total_stopping_time']['n']}, max descent excursion at "
            f"{aggregate['max_excursion']['n']}."
        )
    if run.kernel == GPU_SIEVE_KERNEL:
        return (
            f"Completed full-range descent verification on {run.hardware} for range "
            f"{run.range_start}-{run.range_end}; max descent steps at "
            f"{aggregate['max_total_stopping_time']['n']}, max descent excursion at "
            f"{aggregate['max_excursion']['n']}."
        )
    if run.kernel == CPU_BARINA_KERNEL:
        return (
            f"Completed experimental Barina-domain descent verification on {run.hardware} "
            f"for range {run.range_start}-{run.range_end}; compressed descent record at "
            f"{aggregate['max_total_stopping_time']['n']}."
        )
    return (
        f"Completed {run.kernel} on {run.hardware} for range {run.range_start}-{run.range_end}; "
        f"max total stopping time at {aggregate['max_total_stopping_time']['n']}, "
        f"max excursion at {aggregate['max_excursion']['n']}."
    )


def _validation_mode_label(run: Run, *, range_size: int) -> str:
    base = "full replay" if range_size <= SELECTIVE_VALIDATION_THRESHOLD else "selective"
    if run.kernel == CPU_SIEVE_KERNEL:
        return f"{base}, odd-seed descent reference"
    if run.kernel == GPU_SIEVE_KERNEL:
        return f"{base}, descent reference"
    if run.kernel == CPU_BARINA_KERNEL:
        return f"{base}, experimental Barina audit"
    return base


def execute_run(
    repository: LabRepository,
    run_id: str,
    *,
    checkpoint_interval: int = 250,
) -> Run:
    run = repository.get_run(run_id)
    started_at = run.started_at or utc_now()
    checkpoint = run.checkpoint or {}
    start_at = checkpoint.get("next_value", run.range_start)
    repository.update_run(run.id, status=RunStatus.RUNNING, started_at=started_at)

    aggregate = {
        "processed": 0,
        "last_processed": start_at - 1,
        "max_total_stopping_time": {"n": run.range_start, "value": -1},
        "max_stopping_time": {"n": run.range_start, "value": -1},
        "max_excursion": {"n": run.range_start, "value": -1},
        "sample_records": [],
    }

    if run.metrics:
        aggregate.update(run.metrics)

    writer = _CheckpointWriter()
    try:
        batch_start = start_at
        while batch_start <= run.range_end:
            profile = repository.get_compute_profile()
            effective_checkpoint_interval = _effective_checkpoint_interval(
                run.kernel,
                checkpoint_interval,
                profile=profile,
            )
            batch_end = min(batch_start + effective_checkpoint_interval - 1, run.range_end)
            _t0 = time.perf_counter()
            batch = compute_range_metrics(batch_start, batch_end, kernel=run.kernel, profile=profile)
            _t1 = time.perf_counter()
            aggregate["processed"] += batch.processed
            aggregate["last_processed"] = batch.last_processed
            for key in ("max_total_stopping_time", "max_stopping_time", "max_excursion"):
                if batch.__dict__[key]["value"] > aggregate[key]["value"]:
                    aggregate[key] = batch.__dict__[key]
            aggregate["sample_records"] = (
                aggregate["sample_records"] + batch.sample_records
            )[:12]

            _t2 = time.perf_counter()
            # Snapshot aggregate for the background thread (shallow copy is
            # safe because nested dicts are replaced, never mutated in-place).
            writer.submit(
                repository,
                run.id,
                checkpoint={
                    "next_value": batch_end + 1,
                    "last_processed": batch.last_processed,
                    "checkpoint_interval": effective_checkpoint_interval,
                },
                metrics=dict(aggregate),
                summary=f"Processed {aggregate['processed']} values",
            )
            _t3 = time.perf_counter()
            throttle_sec = _compute_budget_throttle_seconds(
                hardware=run.hardware,
                profile=profile,
                compute_sec=_t1 - _t0,
            )
            if throttle_sec > 0:
                time.sleep(throttle_sec)
            _t4 = time.perf_counter()
            import sys as _sys
            print(json.dumps({
                "timing": "execute_run_batch",
                "kernel": run.kernel,
                "batch_size": batch_end - batch_start + 1,
                "compute_sec": round(_t1 - _t0, 3),
                "aggregate_sec": round(_t2 - _t1, 3),
                "submit_sec": round(_t3 - _t2, 3),
                "throttle_sec": round(throttle_sec, 3),
                "total_sec": round(_t4 - _t0, 3),
                "gpu_pct": round((_t1 - _t0) / (_t4 - _t0) * 100, 1) if (_t4 - _t0) > 0 else 0,
            }), flush=True, file=_sys.stderr)
            batch_start = batch_end + 1

        # Ensure the last in-loop checkpoint is committed before completion.
        writer.shutdown()
    except Exception:
        writer.shutdown()
        raise

    metrics_path = repository.settings.artifacts_dir / "runs" / f"{run.id}.json"
    metrics_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    repository.create_artifact(
        kind=ArtifactKind.JSON,
        path=metrics_path,
        run_id=run.id,
        metadata={
            "direction": run.direction_slug,
            "name": run.name,
            "kernel": run.kernel,
            "hardware": run.hardware,
        },
    )
    checksum = sha256_text(json.dumps(aggregate, sort_keys=True))
    summary = _run_completion_summary(run, aggregate)
    return repository.update_run(
        run.id,
        status=RunStatus.COMPLETED,
        checkpoint={"next_value": run.range_end + 1, "last_processed": run.range_end},
        metrics=aggregate,
        summary=summary,
        checksum=checksum,
        finished_at=utc_now(),
    )


SELECTIVE_VALIDATION_THRESHOLD = 10_000_000
VALIDATION_WINDOW_COUNT = 10
VALIDATION_WINDOW_SIZE = 10_000


def _compute_odd_only_reference(
    start: int, end: int, *, sample_limit: int = 12,
) -> AggregateMetrics:
    """Compute reference aggregate for odd seeds only, using metrics_direct."""
    first_odd = start if start & 1 else start + 1
    max_total: dict = {"n": first_odd, "value": -1}
    max_stopping: dict = {"n": first_odd, "value": -1}
    max_exc: dict = {"n": first_odd, "value": -1}
    sample_records: list[dict] = []
    processed = 0
    for n in range(first_odd, end + 1, 2):
        m = metrics_direct(n)
        processed += 1
        if m.total_stopping_time > max_total["value"]:
            max_total = {"n": n, "value": m.total_stopping_time}
            if len(sample_records) < sample_limit:
                sample_records.append({"metric": "max_total_stopping_time", **max_total})
        if m.stopping_time > max_stopping["value"]:
            max_stopping = {"n": n, "value": m.stopping_time}
            if len(sample_records) < sample_limit:
                sample_records.append({"metric": "max_stopping_time", **max_stopping})
        if m.max_excursion > max_exc["value"]:
            max_exc = {"n": n, "value": m.max_excursion}
            if len(sample_records) < sample_limit:
                sample_records.append({"metric": "max_excursion", **max_exc})
    return AggregateMetrics(
        processed=processed,
        last_processed=end,
        max_total_stopping_time=max_total,
        max_stopping_time=max_stopping,
        max_excursion=max_exc,
        sample_records=sample_records,
    )


def _compute_descent_odd_only_reference(
    start: int, end: int, *, sample_limit: int = 12,
) -> AggregateMetrics:
    """Compute exact standard-Collatz descent reference for odd seeds only."""
    return compute_range_metrics_parallel_descent_odd(
        start,
        end,
        sample_limit=sample_limit,
    )


def _kernel_reference_metrics(run: Run, seed: int) -> NumberMetrics | None:
    if run.kernel in {CPU_DIRECT_KERNEL, CPU_ACCELERATED_KERNEL, CPU_PARALLEL_KERNEL, GPU_KERNEL, CPU_PARALLEL_ODD_KERNEL}:
        return metrics_direct(seed)
    if run.kernel in {CPU_SIEVE_KERNEL, GPU_SIEVE_KERNEL}:
        return metrics_descent_direct(seed)
    if run.kernel == CPU_BARINA_KERNEL:
        return None
    return metrics_direct(seed)


def _reference_aggregate_for_kernel(
    run: Run,
    start: int,
    end: int,
) -> AggregateMetrics | None:
    if run.kernel == CPU_PARALLEL_ODD_KERNEL:
        return _compute_odd_only_reference(start, end)
    if run.kernel == CPU_SIEVE_KERNEL:
        return _compute_descent_odd_only_reference(start, end)
    if run.kernel == GPU_SIEVE_KERNEL:
        return compute_range_metrics_parallel_descent(start, end)
    if run.kernel in {CPU_PARALLEL_KERNEL, GPU_KERNEL}:
        return compute_range_metrics(start, end, kernel=CPU_DIRECT_KERNEL)
    if run.kernel in {CPU_DIRECT_KERNEL, CPU_ACCELERATED_KERNEL}:
        return None
    if run.kernel == CPU_BARINA_KERNEL:
        return None
    return compute_range_metrics(start, end, kernel=CPU_DIRECT_KERNEL)


def _validate_full_replay(run: Run) -> tuple[list[str], list[str]]:
    """Full independent replay used for small runs."""
    mismatches: list[str] = []
    details: list[str] = [
        f"- Mode: full replay ({run.range_end - run.range_start + 1:,} seeds)",
    ]

    if run.kernel in {CPU_DIRECT_KERNEL, CPU_ACCELERATED_KERNEL}:
        for value in range(run.range_start, run.range_end + 1):
            direct = metrics_direct(value)
            accelerated = metrics_accelerated(value)
            if direct != accelerated:
                mismatches.append(
                    f"{value}: direct={asdict(direct)} accelerated={asdict(accelerated)}"
                )
                break
        details.append("- Compared metrics_direct vs metrics_accelerated per-seed")
        return mismatches, details

    if run.kernel == CPU_BARINA_KERNEL:
        mismatches.append(
            "cpu-barina uses compressed domain metrics and is not eligible for "
            "standard value-for-value validation yet."
        )
        details.append(
            "- Validation skipped: cpu-barina is experimental and its step/excursion metrics "
            "are not semantically identical to standard Collatz."
        )
        return mismatches, details

    reference = _reference_aggregate_for_kernel(run, run.range_start, run.range_end)
    candidate = compute_range_metrics(run.range_start, run.range_end, kernel=run.kernel)
    if reference is None:
        mismatches.append(f"No reference aggregate is defined for kernel {run.kernel}.")
    elif _aggregate_validation_payload(reference) != _aggregate_validation_payload(candidate):
        mismatches.append(
            f"aggregate mismatch: reference={_aggregate_validation_payload(reference)} "
            f"candidate={_aggregate_validation_payload(candidate)}"
        )

    if run.kernel == CPU_PARALLEL_ODD_KERNEL:
        details.append(f"- Compared {run.kernel} vs per-seed metrics_direct (odd seeds only)")
    elif run.kernel == CPU_SIEVE_KERNEL:
        details.append(f"- Compared {run.kernel} vs standard-Collatz descent reference (odd seeds only)")
    elif run.kernel == GPU_SIEVE_KERNEL:
        details.append(f"- Compared {run.kernel} vs standard-Collatz descent reference (full range)")
    else:
        details.append(f"- Compared {run.kernel} vs cpu-direct on full range")
    return mismatches, details


def _validate_record_seeds(run: Run) -> tuple[list[str], list[str]]:
    """Verify every record-breaking seed stored in the run's metrics."""
    mismatches: list[str] = []
    details: list[str] = []
    metrics = run.metrics or {}

    if run.kernel == CPU_BARINA_KERNEL:
        details.append("- Record seeds verified: skipped for experimental cpu-barina semantics")
        return mismatches, details

    # Verify the three top-level record holders
    record_keys = {
        "max_total_stopping_time": "total_stopping_time",
        "max_stopping_time": "stopping_time",
        "max_excursion": "max_excursion",
    }
    verified_seeds: set[int] = set()

    def _matches_expected(attr: str, expected_value: int, actual: int) -> bool:
        # Some kernels temporarily cap stored excursion sample records at int64
        # max while retaining the exact top-level record via overflow patches.
        # Validation should treat those capped sample entries as "actual >= cap",
        # not as a hard mismatch.
        if attr == "max_excursion" and expected_value == INT64_MAX:
            return actual >= INT64_MAX
        return actual == expected_value

    for key, attr in record_keys.items():
        record = metrics.get(key)
        if not record or not isinstance(record, dict):
            continue
        n = record.get("n")
        expected_value = record.get("value")
        if n is None or expected_value is None:
            continue
        ref = _kernel_reference_metrics(run, n)
        if ref is None:
            continue
        actual = getattr(ref, attr)
        verified_seeds.add(n)
        if not _matches_expected(attr, expected_value, actual):
            mismatches.append(
                f"Record {key}: seed {n} expected={expected_value} got={actual}"
            )

    # Verify seeds from sample_records
    for record in metrics.get("sample_records", []):
        n = record.get("n")
        metric_name = record.get("metric")
        expected_value = record.get("value")
        if n is None or expected_value is None or n in verified_seeds:
            continue
        ref = _kernel_reference_metrics(run, n)
        if ref is None:
            continue
        verified_seeds.add(n)
        attr = record_keys.get(metric_name)
        if attr is None:
            continue
        actual = getattr(ref, attr)
        if not _matches_expected(attr, expected_value, actual):
            mismatches.append(
                f"Sample record {metric_name}: seed {n} expected={expected_value} got={actual}"
            )

    details.append(f"- Record seeds verified: {len(verified_seeds)}")
    return mismatches, details


def _validate_random_windows(
    run: Run,
    *,
    window_count: int = VALIDATION_WINDOW_COUNT,
    window_size: int = VALIDATION_WINDOW_SIZE,
    rng_seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Sample random windows and compare kernel output vs cpu-direct."""
    mismatches: list[str] = []
    details: list[str] = []
    range_size = run.range_end - run.range_start + 1

    if run.kernel == CPU_BARINA_KERNEL:
        details.append("- Window validation: skipped for experimental cpu-barina semantics")
        mismatches.append(
            "cpu-barina uses compressed domain metrics and is not eligible for "
            "standard window-by-window validation yet."
        )
        return mismatches, details

    def _window_reference(ws: int, we: int) -> AggregateMetrics:
        reference = _reference_aggregate_for_kernel(run, ws, we)
        if reference is None:
            raise ValueError(f"No window reference is defined for kernel {run.kernel}.")
        return reference

    if range_size <= window_size:
        ref = _window_reference(run.range_start, run.range_end)
        cand = compute_range_metrics(run.range_start, run.range_end, kernel=run.kernel)
        if _aggregate_validation_payload(ref) != _aggregate_validation_payload(cand):
            mismatches.append(
                f"Full-range window mismatch: ref={_aggregate_validation_payload(ref)} "
                f"candidate={_aggregate_validation_payload(cand)}"
            )
        details.append(f"- Window validation: 1 window covering full range")
        return mismatches, details

    rng = _random.Random(rng_seed)
    max_window_start = run.range_end - window_size + 1
    windows_checked = 0

    for _ in range(window_count):
        win_start = rng.randint(run.range_start, max(run.range_start, max_window_start))
        win_end = min(win_start + window_size - 1, run.range_end)
        ref = _window_reference(win_start, win_end)
        cand = compute_range_metrics(win_start, win_end, kernel=run.kernel)
        windows_checked += 1
        if _aggregate_validation_payload(ref) != _aggregate_validation_payload(cand):
            mismatches.append(
                f"Window [{win_start:,}, {win_end:,}] mismatch: "
                f"ref={_aggregate_validation_payload(ref)} "
                f"candidate={_aggregate_validation_payload(cand)}"
            )

    details.append(
        f"- Window validation: {windows_checked} random windows of {window_size:,} seeds each"
    )
    return mismatches, details


def _check_verification_coverage_gaps(
    repository: LabRepository,
    run: Run,
) -> tuple[list[str], list[str]]:
    """Check for gaps in the verification coverage up to and including this run."""
    details: list[str] = []
    warnings: list[str] = []

    if run.direction_slug != "verification":
        details.append("- Coverage gap check: skipped (not a verification run)")
        return warnings, details

    all_runs = repository.list_runs()
    covered = [
        r for r in all_runs
        if r.direction_slug == "verification"
        and r.status in {RunStatus.COMPLETED, RunStatus.VALIDATED}
        and r.id != run.id
    ]
    # Include the run being validated
    covered.append(run)
    covered.sort(key=lambda r: (r.range_start, r.range_end))

    # Merge intervals
    merged: list[tuple[int, int]] = []
    for r in covered:
        if merged and r.range_start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], r.range_end))
        else:
            merged.append((r.range_start, r.range_end))

    # Check if coverage starts from 1
    gaps: list[tuple[int, int]] = []
    if merged and merged[0][0] > 1:
        gaps.append((1, merged[0][0] - 1))

    # Check inter-interval gaps
    for i in range(1, len(merged)):
        if merged[i][0] > merged[i - 1][1] + 1:
            gaps.append((merged[i - 1][1] + 1, merged[i][0] - 1))

    if gaps:
        gap_strs = [f"[{g[0]:,}, {g[1]:,}]" for g in gaps[:5]]
        warnings.append(
            f"Coverage gaps detected: {', '.join(gap_strs)}"
            + (f" (+{len(gaps) - 5} more)" if len(gaps) > 5 else "")
        )

    highest = merged[-1][1] if merged else 0
    details.append(
        f"- Coverage: verified up to {highest:,}, "
        f"{len(gaps)} gap(s) found, "
        f"{len(covered)} completed/validated runs"
    )
    return warnings, details


def validate_run(
    repository: LabRepository,
    run_id: str,
    *,
    window_count: int = VALIDATION_WINDOW_COUNT,
    window_size: int = VALIDATION_WINDOW_SIZE,
) -> Run:
    run = repository.get_run(run_id)
    range_size = run.range_end - run.range_start + 1
    mismatches: list[str] = []
    report_details: list[str] = [
        f"- Run: {run.id}",
        f"- Range: [{run.range_start:,}, {run.range_end:,}] ({range_size:,} seeds)",
        f"- Kernel: {run.kernel}",
        f"- Hardware: {run.hardware}",
    ]

    if range_size <= SELECTIVE_VALIDATION_THRESHOLD:
        # ── Full replay for small/bounded runs ──
        m, d = _validate_full_replay(run)
        mismatches.extend(m)
        report_details.extend(d)
    else:
        # ── Selective validation for large runs ──
        report_details.append(
            f"- Mode: selective ({window_count} windows of {window_size:,} + record seeds)"
        )

        # 1. Verify every record-breaking seed independently
        m, d = _validate_record_seeds(run)
        mismatches.extend(m)
        report_details.extend(d)

        # 2. Sample random windows and cross-check kernel vs cpu-direct
        m, d = _validate_random_windows(
            run, window_count=window_count, window_size=window_size,
        )
        mismatches.extend(m)
        report_details.extend(d)

    # 3. Check verification coverage gaps (both small and large runs)
    gap_warnings, gap_details = _check_verification_coverage_gaps(repository, run)
    report_details.extend(gap_details)
    if gap_warnings:
        report_details.append(f"- WARNING: {'; '.join(gap_warnings)}")

    # ── Write validation artifact ──
    validation_path = (
        repository.settings.artifacts_dir / "validations" / f"{run.id}-validation.md"
    )
    if mismatches:
        body = (
            f"# Validation for {run.id}\n\n"
            f"Status: **FAILED**\n\n"
            + "\n".join(report_details) + "\n\n"
            f"## Mismatches\n\n"
            + "\n".join(f"- {m}" for m in mismatches) + "\n"
        )
        validation_path.write_text(body, encoding="utf-8")
        repository.create_artifact(
            kind=ArtifactKind.REPORT,
            path=validation_path,
            run_id=run.id,
            metadata={"status": "failed", "mismatches": len(mismatches)},
        )
        return repository.update_run(
            run.id,
            status=RunStatus.FAILED,
            summary=f"Validation failed: {len(mismatches)} mismatch(es) detected.",
            finished_at=utc_now(),
        )

    mode = _validation_mode_label(run, range_size=range_size)
    body = (
        f"# Validation for {run.id}\n\n"
        f"Status: **PASSED** ({mode})\n\n"
        + "\n".join(report_details) + "\n"
    )
    validation_path.write_text(body, encoding="utf-8")
    repository.create_artifact(
        kind=ArtifactKind.REPORT,
        path=validation_path,
        run_id=run.id,
        metadata={"status": "passed", "mode": mode},
    )
    return repository.update_run(
        run.id,
        status=RunStatus.VALIDATED,
        summary=f"Validation passed ({mode}): independent verification confirmed results.",
        finished_at=utc_now(),
    )


def generate_report(repository: LabRepository) -> Path:
    summary = repository.summary()
    directions = repository.list_directions()
    runs = repository.list_runs()[:10]
    claims = repository.list_claims()[:10]
    report_path = repository.settings.reports_dir / f"lab-report-{utc_now().replace(':', '-')}.md"

    lines = [
        "# Collatz Lab Report",
        "",
        f"- Directions: {summary.direction_count}",
        f"- Runs: {summary.run_count}",
        f"- Validated runs: {summary.validated_run_count}",
        f"- Queued runs: {summary.queued_run_count}",
        f"- Running runs: {summary.running_run_count}",
        f"- Claims: {summary.claim_count}",
        f"- Open tasks: {summary.open_task_count}",
        f"- Artifacts: {summary.artifact_count}",
        f"- Workers: {summary.worker_count}",
        "",
        "## Directions",
        "",
    ]

    for direction in directions:
        lines.extend(
            [
                f"### {direction.title}",
                "",
                f"- Slug: `{direction.slug}`",
                f"- Status: `{direction.status}`",
                f"- Score: `{direction.score}`",
                f"- Success: {direction.success_criteria}",
                f"- Abandon: {direction.abandon_criteria}",
                "",
            ]
        )

    lines.extend(["## Recent Runs", ""])
    for run in runs:
        lines.extend(
            [
                f"- `{run.id}` {run.name}: `{run.status}` on [{run.range_start}, {run.range_end}]",
                f"  Summary: {run.summary}",
            ]
        )

    lines.extend(["", "## Recent Claims", ""])
    for claim in claims:
        lines.extend(
            [
                f"- `{claim.id}` {claim.title}: `{claim.status}`",
                f"  Statement: {claim.statement}",
            ]
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    repository.create_artifact(
        kind=ArtifactKind.REPORT,
        path=report_path,
        metadata={"generated_at": utc_now(), "type": "lab-report"},
    )
    return report_path


def process_next_queued_run(
    repository: LabRepository,
    *,
    worker_id: str,
    supported_hardware: list[str],
    supported_kernels: list[str],
    checkpoint_interval: int = 250,
    validate_after_run: bool = False,
) -> Run | None:
    _ensure_overflow_recovery_runs(repository)
    annotate_legacy_validation_failures(repository)
    run = repository.claim_next_run(
        worker_id=worker_id,
        supported_hardware=supported_hardware,
        supported_kernels=supported_kernels,
    )
    if run is None:
        queued_ids = queue_legacy_validation_reruns(
            repository,
            supported_hardware=supported_hardware,
            limit=4 if "cpu" in supported_hardware else 2,
        )
        if not queued_ids:
            queued_ids = queue_continuous_verification_runs(
                repository,
                supported_hardware=supported_hardware,
            )
        if not queued_ids:
            return None
        run = repository.claim_next_run(
            worker_id=worker_id,
            supported_hardware=supported_hardware,
            supported_kernels=supported_kernels,
        )
        if run is None:
            return None

    try:
        completed_run = execute_run(
            repository,
            run.id,
            checkpoint_interval=checkpoint_interval,
        )
        if validate_after_run:
            completed_run = validate_run(repository, run.id)
        repository.update_worker(worker_id, status="idle", current_run_id=None)
        return completed_run
    except Exception as exc:
        tb_text = _traceback_mod.format_exc()
        logger.error(
            "Run %s failed on worker %s:\n%s", run.id, worker_id, tb_text,
        )
        # Store traceback tail in summary for post-mortem analysis
        short_tb = tb_text[-400:] if len(tb_text) > 400 else tb_text
        failed_run = repository.update_run(
            run.id,
            status=RunStatus.FAILED,
            summary=f"Worker {worker_id} failed: {exc}\n---\n{short_tb}",
            finished_at=utc_now(),
        )
        recovery_ids = _ensure_overflow_recovery_runs(repository, run_ids={run.id})
        if recovery_ids:
            failed_run = repository.update_run(
                run.id,
                summary=(
                    f"Worker {worker_id} failed: {exc} "
                    f"Recovery queued via {', '.join(recovery_ids)}.\n---\n{short_tb}"
                ),
            )
        repository.update_worker(worker_id, status="idle", current_run_id=None)
        return failed_run


def probe_modular_claim(
    *,
    modulus: int,
    allowed_residues: list[int],
    search_limit: int,
) -> ModularProbeResult:
    if modulus < 2:
        raise ValueError("Modulus must be at least 2.")
    if search_limit < 3:
        raise ValueError("Search limit must be at least 3.")

    normalized_residues = sorted({int(value) % modulus for value in allowed_residues})
    checked_odd_values = 0
    counterexamples: list[int] = []

    # Safety bound: the longest known Collatz sequence for seeds below 10^9
    # converges within ~1000 steps.  We use a generous limit that scales with
    # seed magnitude so we never silently skip a seed that actually converges.
    max_steps = max(100_000, search_limit)

    skipped_seeds: list[int] = []
    for start in range(3, search_limit + 1, 2):
        checked_odd_values += 1
        current = start
        steps = 0
        while current != 1 and steps < max_steps:
            current = collatz_step(current)
            steps += 1
        if current != 1:
            skipped_seeds.append(start)
            continue
        if start % modulus not in normalized_residues:
            counterexamples.append(start)
            if len(counterexamples) >= 12:
                break

    first_counterexample = counterexamples[0] if counterexamples else None
    if first_counterexample is None:
        rationale = (
            f"No counterexample was found among odd seeds up to {search_limit} for the residue rule "
            f"mod {modulus} in {normalized_residues}."
        )
    else:
        rationale = (
            f"Found counterexample {first_counterexample}: it reaches 1 in the checked range but its residue "
            f"{first_counterexample % modulus} is outside {normalized_residues} modulo {modulus}."
        )
    if skipped_seeds:
        rationale += (
            f" WARNING: {len(skipped_seeds)} seed(s) did not converge within {max_steps} steps "
            f"and were excluded from analysis (first: {skipped_seeds[0]}). "
            f"These may represent extremely long orbits or potential non-convergence."
        )

    return ModularProbeResult(
        modulus=modulus,
        allowed_residues=normalized_residues,
        checked_limit=search_limit,
        checked_odd_values=checked_odd_values,
        first_counterexample=first_counterexample,
        counterexamples=counterexamples,
        rationale=rationale,
    )
