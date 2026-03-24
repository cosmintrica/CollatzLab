#include <metal_stdlib>
using namespace metal;

// Odd-only descent — mirror ``_collatz_sieve_parallel_odd`` / ``sieve_descent.c``.
//
// Buffer 2: per-seed total steps (int32).  -1 = overflow/cap (needed for overflow_seeds on host).
// Buffer 3: one CollatzChunkPartial per threadgroup — local argmax for steps & max_excursion
//           (avoids int64 max_exc per seed → less device write bandwidth).
//
// Threadgroup width must match Swift (512).

static inline uint ctz_long(ulong x) {
    uint lo = (uint)(x & 0xFFFFFFFFUL);
    return (lo != 0u) ? ctz(lo) : (32u + ctz((uint)(x >> 32)));
}

// Swift host must match this layout: sizeof = stride = 24, alignof = 8
// (int32 + int32 at 0,8; int64 at 8; int32 + int32 at 16,20 — no implicit tail pad).
struct CollatzChunkPartial {
    int32_t max_steps;
    int32_t max_steps_seed_index;
    int64_t max_exc;
    int32_t max_exc_seed_index;
    int32_t _pad;
};

kernel void collatz_lab_sieve_odd(
    constant long &base_odd [[buffer(0)]],
    constant uint &count [[buffer(1)]],
    device int *steps_out [[buffer(2)]],
    device CollatzChunkPartial *partials [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
)
{
    const long COLLATZ_INT64_ODD_STEP_LIMIT = 3074457345618258602L;
    const uint MAX_KERNEL_STEPS = 100000u;
    const uint WG = 512u;

    threadgroup int tg_steps[512];
    threadgroup long tg_exc[512];
    threadgroup uint tg_tid[512];

    int acc_steps = -999999;
    long acc_exc = -1L;

    if (tid < count) {
        long seed = base_odd + 2L * (long)tid;

        if (seed <= 1L) {
            steps_out[tid] = 0;
            acc_steps = 0;
            acc_exc = seed;
        } else {
            long current = seed;
            int steps = 0;
            long max_excursion = current;
            bool failed = false;

            while (current >= seed && steps < (int)MAX_KERNEL_STEPS) {
                if ((current & 1L) == 0L) {
                    current >>= 1;
                    steps++;
                } else {
                    if (current > COLLATZ_INT64_ODD_STEP_LIMIT) {
                        steps_out[tid] = -1;
                        acc_steps = -1;
                        acc_exc = -1L;
                        failed = true;
                        break;
                    }
                    current = 3L * current + 1L;
                    steps++;
                    if (current > max_excursion) {
                        max_excursion = current;
                    }
                    uint tz = ctz_long((ulong)current);
                    long after_all = current >> (int)tz;
                    if (after_all >= seed) {
                        steps += (int)tz;
                        current = after_all;
                    } else {
                        while ((current & 1L) == 0L && current >= seed) {
                            current >>= 1;
                            steps++;
                        }
                    }
                }
            }

            if (!failed) {
                if (current >= seed && current != 1L && steps >= (int)MAX_KERNEL_STEPS) {
                    steps_out[tid] = -1;
                    acc_steps = -1;
                    acc_exc = -1L;
                } else {
                    steps_out[tid] = steps;
                    acc_steps = steps;
                    acc_exc = max_excursion;
                }
            }
        }
    }

    tg_steps[lid] = acc_steps;
    tg_exc[lid] = acc_exc;
    tg_tid[lid] = tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0u) {
        int best_s = -1;
        int best_si = 0;
        long best_e = -1L;
        int best_ei = 0;
        for (uint i = 0u; i < WG; i++) {
            int s = tg_steps[i];
            long e = tg_exc[i];
            uint gtid = tg_tid[i];
            if (s >= 0 && s > best_s) {
                best_s = s;
                best_si = (int)gtid;
            }
            if (s >= 0 && e > best_e) {
                best_e = e;
                best_ei = (int)gtid;
            }
        }
        device CollatzChunkPartial &p = partials[group_id];
        p.max_steps = best_s;
        p.max_steps_seed_index = best_si;
        p.max_exc = best_e;
        p.max_exc_seed_index = best_ei;
        p._pad = 0;
    }
}
