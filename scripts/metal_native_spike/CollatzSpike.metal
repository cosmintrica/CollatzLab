#include <metal_stdlib>
using namespace metal;

// Exploratory kernel: odd-only descent until first drop below seed (sieve-style contract).
// One thread per odd seed: seed = base_odd + 2 * tid
kernel void collatz_odd_descent_spike(
    constant ulong &base_odd [[buffer(0)]],
    constant uint &count [[buffer(1)]],
    device uint *steps_out [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
)
{
    if (tid >= count) {
        return;
    }
    ulong seed = base_odd + 2ul * (ulong)tid;
    ulong n = seed;
    uint steps = 0u;
    const uint MAX_STEPS = 100000u;
    // Largest odd n such that 3*n+1 fits in uint64
    const ulong ODD_LIMIT = 6148914691236517204ul;

    while (steps < MAX_STEPS) {
        if (n == 0ul || n == 1ul) {
            break;
        }
        if (n < seed) {
            break;
        }
        if ((n & 1ul) == 0ul) {
            n /= 2ul;
        } else {
            if (n > ODD_LIMIT) {
                steps = 0xFFFFFFFFu;
                break;
            }
            n = 3ul * n + 1ul;
        }
        steps++;
    }
    steps_out[tid] = steps;
}
