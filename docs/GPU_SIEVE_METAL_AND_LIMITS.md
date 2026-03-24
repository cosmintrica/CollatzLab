# Native Metal `gpu-sieve` — behaviour, limits, next steps

## What ships today

- **Modular:** `compute_range_metrics_gpu_sieve` on **macOS** may call `gpu_sieve_metal_runtime.compute_range_metrics_gpu_sieve_metal` when `COLLATZ_GPU_SIEVE_BACKEND` allows it. **Linux / Windows** never execute this path; they keep **CUDA** or **no GPU** as before.
- **Diagnostics:** `GET /api/workers/gpu-sieve-metal` — helper path, ping, stdio. Combined with native CPU sieve: `GET /api/workers/native-stack` (see [`MACOS_FULL_NATIVE_CPU_AND_METAL_GPU.md`](./MACOS_FULL_NATIVE_CPU_AND_METAL_GPU.md)).
- **Automatic detection (`auto`):** if `metal_sieve_chunk` is present and **`--ping` succeeds** (exit 0), Metal is used; otherwise **PyTorch MPS**. **`--ping` loads the Metal library and compute pipeline** and asserts `CollatzChunkPartial` Swift/Metal ABI (24-byte stride); exit **1** if the helper or `.metallib` is broken — so “ping” really means “Metal path is viable.” If Metal fails at runtime, **`auto` falls back to MPS** (logged).
- **Persistent helper (`--stdio`):** successful `--ping` returns `{"ok":true,"stdio":true,"metal_abi_ok":true}`. Python keeps **one** `metal_sieve_chunk --stdio` process alive and sends one JSON line per chunk (`first_odd`, `count`), receiving one JSON line back — **much lower overhead** than spawning a process per chunk. Disable with `COLLATZ_METAL_SIEVE_STDIO=0` to force one-shot subprocess mode (compatible with older binaries that only returned `{"ok":true}` without `metal_abi_ok`).
- **RAM after compute:** the helper’s Metal `steps` buffer is **grow-only** (it sizes up to the largest odd-count chunk you have run). **Stopping a run in the UI does not kill that process** while the API/worker Python process stays up — Activity Monitor can still show multi‑GB `metal_sieve_chunk` until the child exits. Mitigations: **(1)** chunk benchmarks end with `shutdown_metal_stdio_transport()`; **(2)** optional `COLLATZ_METAL_SIEVE_STDIO_SHUTDOWN_AFTER_RUN=1` so `execute_run` tears down the helper after **each** lab run (frees RAM; default **off** to avoid races with back-to-back GPU jobs); **(3)** manual `POST /api/workers/gpu-sieve-metal/stdio-shutdown`; **(4)** restart uvicorn/worker; **(5)** `COLLATZ_METAL_SIEVE_STDIO=0` trades speed for a fresh process per chunk.
- **Logs:** `Metal stdio chunk failed; falling back to one-shot subprocess` means the persistent child died or closed pipes while Python still expected a line on stdout (often right after an aggressive shutdown or a crashed helper). The message now includes **returncode** and any **stderr** captured. Rebuild `metal_sieve_chunk` after Swift changes; avoid toggling `COLLATZ_METAL_SIEVE_STDIO_SHUTDOWN_AFTER_RUN` on busy workers unless you accept extra process churn.
- **Headless Mac without PyTorch:** set `COLLATZ_GPU_SIEVE_METAL_WITHOUT_TORCH=1` so `gpu-sieve` can run when **native Metal** is available but **MPS is not** (worker discovery injects `gpu-sieve` on Apple GPUs in that mode). **`gpu-collatz-accelerated`** still needs CUDA/MPS.
- **Parity:** Metal uses the same odd-descent loop as Numba (`_collatz_sieve_parallel_odd`). Overflow seeds (`steps == -1`) are patched with `metrics_descent_direct`, matching the MPS path.
- **Streaming:** each chunk processes up to **N** odd seeds (chunk size). The kernel still writes **per-seed `steps`** (for `overflow_seeds`), but **max excursion** is reduced **per threadgroup** into a small `CollatzChunkPartial` buffer; Swift merges **O(threadgroups)** partials instead of reading **N × int64** max-exc values.
- **`CollatzChunkPartial` ABI:** Metal struct is C-layout: two `int32`, one `int64`, two `int32` → **24 bytes**, **8-byte alignment**. Swift asserts in `MetalSieveChunkEngine.init` (runs on **`--ping`**, `--stdio`, and every chunk). The standalone tool is **`metal_lab_sieve_verify`** (`build_metal_verify.sh`); it also creates symlink **`metal_sieve_verify`** → same binary for shorter docs/commands.

## Limits (why they exist)

| Limit | Reason |
|-------|--------|
| **Subprocess per chunk** | No in-process Metal API from Python without a compiled extension. Spawning `metal_sieve_chunk` adds **milliseconds** of overhead per chunk; huge chunk sizes amortise this but very small chunks hurt. |
| **macOS-only** | Metal is Apple’s API; other platforms are out of scope for this helper. |
| **Binary must exist** | Runtime does not compile Swift/Metal. CI / users must **build once** (`build_metal_sieve_chunk.sh`) or ship a **prebuilt** binary + `.metallib`. |
| **PyTorch vs Metal-only** | By default **`gpu_execution_ready`** still wants MPS/CUDA. With **`COLLATZ_GPU_SIEVE_METAL_WITHOUT_TORCH=1`** and a built `metal_sieve_chunk`, **`gpu-sieve` alone** can run without PyTorch; other GPU kernels still need PyTorch/CUDA. |
| **Reference parity tests skip without MPS** | Integration test `test_metal_gpu_sieve_streaming_matches_mps` compares Metal vs MPS; it **skips** if `torch.backends.mps` is unavailable (e.g. Linux CI). |
| **Overflow patch** | Same rare `int64` edge cases as MPS; true excursion &gt; `INT64_MAX` is carried via `_OverflowPatch` after aggregation. |

## CPU native (C) vs Numba

- **Script:** `scripts/native_sieve_kit/benchmark_cpu_native_vs_numba.py` — wall time for **`sieve_descent bench`** vs **`compute_range_metrics_sieve_odd`** on the same odd count.
- **Correctness:** `compare_native_cpu_vs_lab.py` and pytest (`test_sieve_reference_vs_numba`, `test_native_sieve_c_aggregate`) — **not** Apple-Silicon-specific; needs `cc`.

## Next technical steps (optional)

1. ~~**Reduce subprocess overhead:**~~ **Done:** `--stdio` persistent process (see above). **Done (overlap):** optional pipelined stdin reader while the GPU runs the current chunk (`COLLATZ_METAL_SIEVE_STDIO_PIPELINE`, default on). Further: **`.dylib` + ctypes** — see [`METAL_DYLIB_ROADMAP.md`](./METAL_DYLIB_ROADMAP.md).
2. **Packaging:** vend **prebuilt** `metal_sieve_chunk` + `CollatzLabSieve.metallib` for tagged releases; document codesign / notarisation if distributed outside the repo.
3. **GPU-ready without PyTorch:** separate capability flag for “Metal sieve only” if you want headless Mac agents without CUDA/MPS for other kernels.
4. ~~**On-device reduction:**~~ **Done (partial):** per-threadgroup argmax for stopping + excursion in `CollatzLabSieve.metal`; per-seed `steps` kept for overflow detection. Further win would require a second pass or atomics to drop the `steps` array too.
5. **CI:** macOS runner job: build Metal helper + run `test_metal_gpu_sieve_integration` (requires PyTorch MPS on the runner).

See also [`research/NATIVE_SIEVE_PORT.md`](../research/NATIVE_SIEVE_PORT.md).
