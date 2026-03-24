# Native (C) `cpu-sieve` backend — setup and correctness

## Behaviour

- **Default:** `COLLATZ_CPU_SIEVE_BACKEND=auto` (or unset) — if `libsieve_descent_native.{dylib|so}` is found on the resolved search paths, the **native** library is used; otherwise **Numba**.
- **Override:** `numba` = always the parallel Numba kernel; `native` = always the C shared library — raises **`ValueError`** if the library is missing (no silent fallback).
- **Native:** fills NumPy arrays from C (`collatz_lab_cpu_sieve_odd_fill`), then the **same** Python code as Numba for:
  - overflow patching (`metrics_descent_direct`, `_OverflowPatch`);
  - aggregates + `sample_records` (including the same argmax linkage for total/stopping as before).

The C source is `scripts/native_sieve_kit/sieve_descent.c` (same loop as `odd_sieve_descent_one` used by `verify` / `bench`).

## Build

**macOS (recommended for best performance):** install OpenMP from Homebrew, then build:

```bash
brew install libomp
cd CollatzLab
bash scripts/native_sieve_kit/build_native_cpu_sieve_lib.sh
```

The script tries **OpenMP** first (Darwin: `libomp` + `rpath` to `libomp.dylib`; Linux: `-fopenmp`). If linking fails, it falls back to a **sequential** C loop.

```bash
# or: bash scripts/native_sieve_kit/build_c.sh  (builds the binary and the library on Darwin/Linux)
```

Output: `scripts/native_sieve_kit/libsieve_descent_native.dylib` (macOS) or `.so` (Linux).

The symbol `collatz_lab_cpu_sieve_build_info()` returns `1` if the library was linked with OpenMP, `0` for a sequential build (used in diagnostics).

**Windows:** the kit does not ship an official `.dll` build script; stay on Numba or compile an equivalent export manually.

## OpenMP in brief

**OpenMP** is a standard API (pragmas + runtime) for **CPU parallelism** in C/C++/Fortran: the compiler generates a thread pool and splits `for` loop iterations across threads. In `collatz_lab_cpu_sieve_odd_fill`, each index `i` only writes `out_*[i]` → **no data races**; results are **deterministic** and match the sequential loop (parity covered by tests).

OpenMP thread count is usually controlled with `OMP_NUM_THREADS` (and related variables), not the Numba compute profile.

### macOS: two `libomp` copies → crash at import (PyTorch + native cpu-sieve)

If the native library was linked against **Homebrew** `libomp.dylib` and the same Python process also loads **PyTorch** (which bundles another `libomp`), the second runtime can abort during startup with **OMP: Error #15** / `__kmp_register_library_startup` → `SIGABRT`.

**Mitigation (default in this repo):** `KMP_DUPLICATE_LIB_OK=TRUE` is set from `collatz_lab.runtime_bootstrap` when entering `lab` CLI and `collatz_lab.main` (API), and exported by `scripts/mac-dev-stack.sh`, `run-worker.sh`, and `run-backend.sh`. It is an LLVM-documented, unsupported workaround; unset or set to `FALSE` only if you intentionally want the process to fail fast instead of continuing.

**Also:** run workers/API with **`./.venv/bin/python`**, not Homebrew’s bare `python3`, so the environment matches the stack scripts (Cursor tasks sometimes pick the wrong interpreter).

## Environment variables

| Variable | Role |
|----------|------|
| `COLLATZ_CPU_SIEVE_BACKEND` | `auto` (default), `numba`, or `native`. |
| `COLLATZ_CPU_SIEVE_NATIVE_LIB` | Optional absolute path to the shared library. |
| `COLLATZ_LAB_ROOT` | Repository root for default kit path resolution. |
| `OMP_NUM_THREADS` | (OpenMP builds only) caps threads used in `collatz_lab_cpu_sieve_odd_fill`. |
| `KMP_DUPLICATE_LIB_OK` | macOS: set to `TRUE` when PyTorch and Homebrew-linked native cpu-sieve share a process (see above). |

## Performance (local measurements)

Throughput depends on the machine, Numba JIT, `NUMBA_NUM_THREADS` / worker profile, and OpenMP. For a quick comparison on the same interval:

```bash
python3 scripts/native_sieve_kit/benchmark_cpu_sieve_numba_vs_native.py --start 1 --end 500000
```

Before timing, the script runs a **warmup** on `[1, min(100_000, end)]` when `end > 20_000` so **`seconds_numba` does not include Numba JIT compile time**.

The JSON output includes `seconds_numba`, `seconds_resolved`, `resolved_backend`, `openmp_linked`, `warmup_range_applied`, and speed ratios. **There is no universal single number** — run on your hardware after `brew install libomp` and a rebuild.

## API

`GET /api/workers/cpu-sieve-native` — metadata: path, `available`, `backend_mode`, `resolved_backend`, `openmp_linked`, `resolved_error` (when `native` is forced but the library is missing).

`GET /api/workers/native-stack` — the same CPU block plus Metal `gpu-sieve` diagnostics in one JSON (see [`MACOS_FULL_NATIVE_CPU_AND_METAL_GPU.md`](./MACOS_FULL_NATIVE_CPU_AND_METAL_GPU.md)).

## Tests

`backend/tests/test_cpu_sieve_native_parity.py` — compares validation aggregates for native vs Numba on safe ranges; also covers `auto` when the library is present.

See also [`CPU_NATIVE_VS_NUMBA.md`](./CPU_NATIVE_VS_NUMBA.md) and [`research/NATIVE_SIEVE_PORT.md`](../research/NATIVE_SIEVE_PORT.md).
