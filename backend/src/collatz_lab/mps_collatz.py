"""
Apple Silicon GPU path for ``gpu-collatz-accelerated`` and ``gpu-sieve`` using PyTorch MPS.

- Accelerated: semantics match ``_collatz_metrics_parallel`` (full orbit to 1).
- Sieve (odd-only descent): semantics match ``_collatz_sieve_parallel_odd`` in ``services``.
"""

from __future__ import annotations

import os

import numpy as np

INT64_MAX = 2**63 - 1
COLLATZ_INT64_ODD_STEP_LIMIT = (INT64_MAX - 1) // 3
MAX_KERNEL_STEPS = 100_000
MAX_HALVE_TAIL = 72

# Each ``tensor.any()`` on MPS forces a GPU→CPU sync. Default: re-check only every N outer steps.
def _mps_sync_every() -> int:
    try:
        v = int(os.getenv("COLLATZ_MPS_SYNC_EVERY", "128"))
    except ValueError:
        v = 128
    return max(1, min(2048, v))


def mps_accelerated_available() -> bool:
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def compute_accelerated_arrays_mps(first: int, size: int, *, batch_size: int = 32768) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    device = torch.device("mps")
    total_out = np.empty(size, dtype=np.int32)
    stopping_out = np.empty(size, dtype=np.int32)
    max_exc_out = np.empty(size, dtype=np.int64)

    offset = 0
    while offset < size:
        chunk = min(batch_size, size - offset)
        seeds = torch.arange(first + offset, first + offset + chunk, dtype=torch.int64, device=device)
        total, stopping, max_exc = _mps_batch_accelerated(seeds)
        total_out[offset : offset + chunk] = total.cpu().numpy()
        stopping_out[offset : offset + chunk] = stopping.cpu().numpy()
        max_exc_out[offset : offset + chunk] = max_exc.cpu().numpy()
        offset += chunk

    return total_out, stopping_out, max_exc_out


def _mps_batch_accelerated(seeds: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    import torch

    device = seeds.device
    n = seeds.shape[0]
    sync_every = _mps_sync_every()
    m1i32 = torch.tensor(-1, dtype=torch.int32, device=device)
    m1i64 = torch.tensor(-1, dtype=torch.int64, device=device)
    z_i32 = torch.tensor(0, dtype=torch.int32, device=device)

    current = seeds.clone()
    original = seeds.clone()
    steps = torch.zeros(n, dtype=torch.int32, device=device)
    stopping_time = torch.full((n,), -1, dtype=torch.int32, device=device)
    max_ex = current.clone()
    overflow = torch.zeros(n, dtype=torch.bool, device=device)

    for _iter in range(MAX_KERNEL_STEPS):
        not_done = (current != 1) & ~overflow & (current >= 0)

        cur = current
        even_mask = not_done & (cur % 2 == 0)
        nxt = cur // 2
        steps = steps + even_mask.to(torch.int32)
        cross = even_mask & (nxt < original) & (stopping_time < 0)
        stopping_time = torch.where(cross, steps, stopping_time)
        max_ex = torch.where(even_mask, torch.maximum(max_ex, torch.maximum(cur, nxt)), max_ex)
        current = torch.where(even_mask, nxt, current)

        not_done = (current != 1) & ~overflow & (current >= 0)
        odd_mask = not_done & (current % 2 != 0)

        bad = odd_mask & (current > COLLATZ_INT64_ODD_STEP_LIMIT)
        overflow = overflow | bad
        steps = torch.where(bad, m1i32, steps)
        stopping_time = torch.where(bad, m1i32, stopping_time)
        max_ex = torch.where(bad, m1i64, max_ex)
        current = torch.where(bad, m1i64, current)

        odd_mask = (current != 1) & ~overflow & (current >= 0) & (current % 2 != 0)
        cur = current
        nxt = 3 * cur + 1
        steps = steps + odd_mask.to(torch.int32)
        cross = odd_mask & (nxt < original) & (stopping_time < 0)
        stopping_time = torch.where(cross, steps, stopping_time)
        max_ex = torch.where(odd_mask, torch.maximum(max_ex, torch.maximum(cur, nxt)), max_ex)
        current = torch.where(odd_mask, nxt, current)

        trail_odd = odd_mask
        for __ in range(MAX_HALVE_TAIL):
            t = trail_odd & (current % 2 == 0) & (current != 1) & ~overflow & (current >= 0)
            nxt2 = current // 2
            steps = steps + t.to(torch.int32)
            cross2 = t & (nxt2 < original) & (stopping_time < 0)
            stopping_time = torch.where(cross2, steps, stopping_time)
            max_ex = torch.where(t, torch.maximum(max_ex, torch.maximum(current, nxt2)), max_ex)
            current = torch.where(t, nxt2, current)

        if (_iter + 1) % sync_every == 0:
            not_done = (current != 1) & ~overflow & (current >= 0)
            if not bool(not_done.any().item()):
                break

        if _iter + 1 >= MAX_KERNEL_STEPS:
            still = (current != 1) & ~overflow & (current >= 0)
            overflow = overflow | still
            steps = torch.where(still, m1i32, steps)
            stopping_time = torch.where(still, m1i32, stopping_time)
            max_ex = torch.where(still, m1i64, max_ex)
            break

    final_steps = torch.where(overflow, m1i32, steps)
    final_stopping = torch.where(stopping_time >= 0, stopping_time, steps)
    final_stopping = torch.where(overflow, m1i32, final_stopping)

    one_mask = seeds <= 1
    final_steps = torch.where(one_mask, z_i32, final_steps)
    final_stopping = torch.where(one_mask, z_i32, final_stopping)
    max_ex = torch.where(one_mask, torch.maximum(seeds, torch.tensor(1, dtype=torch.int64, device=device)), max_ex)

    return final_steps, final_stopping, max_ex


def compute_sieve_arrays_mps(
    first_odd: int,
    odd_count: int,
    *,
    batch_size: int = 2_097_152,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Odd-only descent on MPS; returns **full** per-seed arrays (host RAM = O(odd_count)).

    For production ``gpu-sieve`` checkpoints, use streaming reduction in
    ``services._compute_range_metrics_gpu_sieve_mps`` instead — large ``odd_count``
    here will allocate gigabytes and thrash memory.
    """
    import torch

    device = torch.device("mps")
    total_out = np.empty(odd_count, dtype=np.int32)
    stopping_out = np.empty(odd_count, dtype=np.int32)
    max_exc_out = np.empty(odd_count, dtype=np.int64)

    fo = torch.tensor(int(first_odd), dtype=torch.int64, device=device)
    offset = 0
    while offset < odd_count:
        chunk = min(batch_size, odd_count - offset)
        idx = torch.arange(offset, offset + chunk, dtype=torch.int64, device=device)
        originals = fo + 2 * idx
        total, stopping, max_exc = _mps_batch_sieve_descent(originals)
        total_out[offset : offset + chunk] = total.cpu().numpy()
        stopping_out[offset : offset + chunk] = stopping.cpu().numpy()
        max_exc_out[offset : offset + chunk] = max_exc.cpu().numpy()
        offset += chunk

    return total_out, stopping_out, max_exc_out


def _mps_batch_sieve_descent(originals: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Vectorized odd-only descent (same control flow as Numba ``_collatz_sieve_parallel_odd``)."""
    import torch

    device = originals.device
    n = originals.shape[0]
    sync_every = _mps_sync_every()
    m1i32 = torch.tensor(-1, dtype=torch.int32, device=device)
    m1i64 = torch.tensor(-1, dtype=torch.int64, device=device)
    z_i32 = torch.tensor(0, dtype=torch.int32, device=device)

    current = originals.clone()
    steps = torch.zeros(n, dtype=torch.int32, device=device)
    max_exc = originals.clone().to(torch.int64)
    overflow = torch.zeros(n, dtype=torch.bool, device=device)
    seed_le1 = originals <= 1

    for _iter in range(MAX_KERNEL_STEPS):
        active = ~overflow & ~seed_le1 & (current >= originals) & (current > 0)

        even_m = active & (current % 2 == 0)
        nxt = current // 2
        steps = steps + even_m.to(torch.int32)
        max_exc = torch.where(even_m, torch.maximum(max_exc, torch.maximum(current, nxt)), max_exc)
        current = torch.where(even_m, nxt, current)

        odd_m = ~overflow & ~seed_le1 & (current >= originals) & (current > 0) & (current % 2 != 0)

        bad = odd_m & (current > COLLATZ_INT64_ODD_STEP_LIMIT)
        overflow = overflow | bad
        steps = torch.where(bad, m1i32, steps)
        max_exc = torch.where(bad, m1i64, max_exc)
        current = torch.where(bad, m1i64, current)

        odd_m = ~overflow & ~seed_le1 & (current >= originals) & (current > 0) & (current % 2 != 0)
        cur = current
        nxt = 3 * cur + 1
        steps = steps + odd_m.to(torch.int32)
        max_exc = torch.where(odd_m, torch.maximum(max_exc, torch.maximum(cur, nxt)), max_exc)
        current = torch.where(odd_m, nxt, current)

        trail_odd = odd_m
        for __ in range(MAX_HALVE_TAIL):
            t = trail_odd & ~overflow & ~seed_le1 & (current >= originals) & (current > 0) & (current % 2 == 0)
            nxt2 = current // 2
            steps = steps + t.to(torch.int32)
            max_exc = torch.where(t, torch.maximum(max_exc, torch.maximum(current, nxt2)), max_exc)
            current = torch.where(t, nxt2, current)

        if (_iter + 1) % sync_every == 0:
            active = ~overflow & ~seed_le1 & (current >= originals) & (current > 0)
            if not bool(active.any().item()):
                break

        if _iter + 1 >= MAX_KERNEL_STEPS:
            still = ~overflow & ~seed_le1 & (current >= originals) & (current != 1)
            overflow = overflow | still
            steps = torch.where(still, m1i32, steps)
            max_exc = torch.where(still, m1i64, max_exc)
            break

    one_i64 = torch.tensor(1, dtype=torch.int64, device=device)
    total = torch.where(seed_le1, z_i32, torch.where(overflow, m1i32, steps))
    stopping = total.clone()
    max_out = torch.where(
        seed_le1,
        torch.maximum(originals, one_i64),
        torch.where(overflow, m1i64, max_exc),
    )
    return total, stopping, max_out
