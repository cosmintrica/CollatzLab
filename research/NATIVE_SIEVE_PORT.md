# Native sieve port (Metal GPU + native CPU) — plan

## Goals

1. **GPU (`gpu-sieve`)** — optional backend that runs the **same odd-descent contract** as PyTorch MPS, with **streaming aggregates** and SQLite checkpoints (no O(N) host buffers for huge runs).
2. **CPU** — optional **native C** (or static lib) path to compare against **Numba** for throughput and **bit-identical aggregates** on safe ranges.

## Correctness spine (implemented first)

| Layer | Location | Role |
|-------|----------|------|
| Python reference | `backend/src/collatz_lab/sieve_reference.py` | Canonical mirror of `_collatz_sieve_parallel_odd` (slow). |
| Tests | `backend/tests/test_sieve_reference_vs_numba.py` | Ref vs **cpu-sieve** on small linear ranges. |
| Native C | `scripts/native_sieve_kit/sieve_descent.c` | Same loop as Numba; `verify` prints aggregate JSON. |
| Compare script | `scripts/native_sieve_kit/compare_native_cpu_vs_lab.py` | C vs ref vs Numba. |
| Metal kernel | `scripts/native_sieve_kit/metal/CollatzLabSieve.metal` | GPU implementation of the same loop; `verify_main.swift` reduces on CPU. |

**Overflow:** Numba may return `-1` then **patch** with `metrics_descent_direct`. The reference and first native ports intentionally target ranges **without** `-1` for strict equality tests.

## Build / run (developer)

```bash
bash scripts/native_sieve_kit/build_c.sh
PYTHONPATH=backend/src python3 scripts/native_sieve_kit/compare_native_cpu_vs_lab.py

bash scripts/native_sieve_kit/metal/build_metal_verify.sh
./scripts/native_sieve_kit/metal/metal_lab_sieve_verify --base 1 --count 5000
```

Compare Metal JSON line to `odd_sieve_descent_linear_range(1, 1 + 2*(5000-1))` manually or extend the Python script.

## Phase B — integrate `gpu-sieve` Metal into the lab

**Done (initial):** `collatz_lab.gpu_sieve_metal_runtime` + `metal_sieve_chunk` subprocess, `COLLATZ_GPU_SIEVE_BACKEND=auto|mps|metal`, streaming merge + overflow patch, `GET /api/workers/gpu-sieve-metal`. See [`docs/GPU_SIEVE_METAL_AND_LIMITS.md`](../docs/GPU_SIEVE_METAL_AND_LIMITS.md).

**Still open:**

1. **Lower IPC overhead** (long-lived process or dylib).
2. **macOS CI** job with PyTorch MPS + Metal build to run `test_metal_gpu_sieve_integration` non-skip.
3. **Prebuilt binaries** in releases (no Xcode on end-user machines).

## Phase C — optional native CPU in worker

**Shipped:** default `COLLATZ_CPU_SIEVE_BACKEND=auto` picks `libsieve_descent_native.{dylib|so}` when present (else Numba); `native` / `numba` force a path. Build script tries **OpenMP** when `libomp` (Darwin) or `-fopenmp` (Linux) works, else sequential. Python applies the **same** overflow patch and aggregate reduction as the Numba path (`_cpu_sieve_odd_finalize_from_arrays`). Tests: `test_cpu_sieve_native_parity.py`. Docs: [`docs/CPU_SIEVE_NATIVE_BACKEND.md`](../docs/CPU_SIEVE_NATIVE_BACKEND.md). Benchmark: `scripts/native_sieve_kit/benchmark_cpu_sieve_numba_vs_native.py`.

**Still open:** tuning (chunk size / scheduling), optional static link or packaging without Homebrew `libomp` on macOS.

## Related docs

- [`docs/PERFORMANCE_MACOS.md`](../docs/PERFORMANCE_MACOS.md)
- [`docs/RUN_RECOVERY.md`](../docs/RUN_RECOVERY.md)
