# Comparison: native CPU (C) vs Numba (`cpu-sieve`)

## What we compare

| Path | What it measures | Contract |
|------|------------------|----------|
| **C** (`sieve_descent bench`) | Wall time over **N consecutive odd** seeds, standard Collatz descent (lab sieve–style metrics). | Implementation in `scripts/native_sieve_kit/sieve_descent.c`. |
| **Numba** | `compute_range_metrics_sieve_odd(1, last_linear)` — same set of **N odds** (linear range `[1, 1+2(N-1)]`). | Production code in `collatz_lab.services`. |

This is **not** Metal or GPU; it is **CPU x86_64/ARM** for parity and benchmarking.

## Correctness (before interpreting speed)

- `scripts/native_sieve_kit/compare_native_cpu_vs_lab.py` — C vs lab aggregates on a small/medium interval.
- pytest: `test_sieve_reference_vs_numba`, `test_native_sieve_c_aggregate`.

## Numbers (how to reproduce)

From the repository root:

```bash
cd CollatzLab
PYTHONPATH=backend/src .venv/bin/python scripts/native_sieve_kit/benchmark_cpu_native_vs_numba.py --odd-count 500000
```

**JSON** output:

- `native_c_wall_s` — wall time for the C binary (`bench`).
- `numba_cpu_sieve_wall_s` — wall time for `compute_range_metrics_sieve_odd`.
- **`numba_vs_native_c_ratio`** = `numba_wall / c_wall` → usually **> 1** (Numba slower than C `-O3` on the same chip).

The exact ratio **depends on CPU, LLVM/Numba version, and system load**; there is no single “absolute truth,” only **the ratio measured on your machine**.

### How to read an example

If `numba_vs_native_c_ratio` ≈ **47**, Numba was ~**47× slower** than the C binary for that run on the **same odd-count** (total wall time, not per-seed in isolation).

**Reference run (example, Mac dev, `--odd-count 200000`):** `numba_vs_native_c_ratio` ≈ **51** (Numba ~51× wall time vs C `-O3` on the same odd interval). Repeat on your hardware — values vary with CPU, load, and versions.

## Why C can win

- No JIT overhead on first call (the Numba benchmark order matters in practice; the script measures **one** invocation each).
- `-O3` and no GIL — the compiler can optimise the hot loop aggressively.
- Numba adds **`prange` scheduling**, checks, and its own memory model.

## Link to `cpu-sieve` in the worker

- **Default:** `COLLATZ_CPU_SIEVE_BACKEND=auto` — **native** if `libsieve_descent_native` is present, else **Numba** (same aggregate/overflow contract).
- **Force:** `numba` or `native` (no fallback if the library is missing when `native` is forced). OpenMP build on macOS: `brew install libomp` + `build_native_cpu_sieve_lib.sh` (see [`CPU_SIEVE_NATIVE_BACKEND.md`](./CPU_SIEVE_NATIVE_BACKEND.md)).
- **End-to-end measurement** (Numba JIT vs `.dylib` + same Python finalize): `scripts/native_sieve_kit/benchmark_cpu_sieve_numba_vs_native.py`.
