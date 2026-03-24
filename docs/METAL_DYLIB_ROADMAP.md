# Roadmap: Metal `gpu-sieve` via `.dylib` + ctypes (no separate process / JSON)

## Motivation

`metal_sieve_chunk --stdio` removes spawn-per-chunk, but you still have:

- JSON framing + parsing on every chunk;
- two processes (Python + helper) and fragmented memory accounting in Activity Monitor;
- latency on each request→response round.

A **dynamic library** loaded from Python (`ctypes` or a minimal C extension) could expose:

```c
// sketch API (not in repo yet)
int collatz_metal_sieve_chunk_init(const char* metallib_dir);
void collatz_metal_sieve_chunk_shutdown(void);
int collatz_metal_sieve_chunk_run(
    int64_t first_odd,
    uint32_t odd_count,
    int32_t* out_max_steps,
    int64_t* out_max_steps_seed,
    int64_t* out_max_exc,
    int64_t* out_max_exc_seed,
    int32_t* out_steps,          // or NULL + streaming overflow list
    size_t steps_stride,
    int64_t* overflow_out,
    size_t* overflow_count
);
```

## Technical steps (recommended order)

1. **Extract the Metal engine** from `chunk_main.swift` into a **framework** or **dylib** target (Swift + `@_cdecl` thin wrappers, or **Objective-C++** / **C++** with `MTL*` directly).
2. **Stabilize the C ABI** (header `collatz_metal_sieve.h`) — no Swift types on the surface.
3. **Python load**: `ctypes.CDLL("libcollatz_metal_sieve.dylib")` + `restype`/`argtypes`; alternatively **cffi** / **maturin** (PyO3) if you want type-safety.
4. **Parity**: same tests as `test_metal_gpu_sieve_integration.py` — compare aggregates vs `cpu-sieve` / MPS.
5. **Packaging**: codesign + `rpath` to `CollatzLabSieve.metallib` next to the dylib; document in `MACOS_FULL_NATIVE_CPU_AND_METAL_GPU.md`.

## Cost / risk

- **High:** stable Swift→C bridging, device/command-queue lifecycle, Metal errors through ctypes.
- **Benefit:** removes JSON and (optionally) enables **double-buffering** in-process without stdio.

Until then, use **`COLLATZ_METAL_SIEVE_STDIO_PIPELINE=1`** (overlap stdin read / GPU), **`COLLATZ_METAL_SIEVE_CHUNK_AUTO=1`** (default on macOS when chunk size is unset), and tune **`scripts/profile_metal_sieve_chunk.py`**.

**Estimate without dylib:** `scripts/benchmark_metal_stdio_overhead.py` measures median time for very small vs large chunks and gives a **heuristic** for fixed stdio overhead per round — it does not replace a full benchmark on your verification interval.
