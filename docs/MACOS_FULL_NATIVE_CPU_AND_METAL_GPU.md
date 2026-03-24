# macOS: native CPU (OpenMP) + GPU Metal — full setup

Correctness and **source-of-truth** for the whole lab (all OSes and backends) live in [`CORRECTNESS_AND_VALIDATION.md`](./CORRECTNESS_AND_VALIDATION.md) and `collatz_lab.validation_source` — this page is only the **macOS deployment** for two of the fast paths.

This guide wires up **both** fast paths in CollatzLab on Apple Silicon (or Intel Mac with Metal):

| Run | Goal | Requirement |
|-----|------|-------------|
| **CPU worker**, kernel `cpu-sieve` | `libsieve_descent_native.dylib` + OpenMP | `brew install libomp`, then build |
| **GPU worker**, kernel `gpu-sieve` | `metal_sieve_chunk` | Xcode Command Line Tools (`xcrun`, `swiftc`) |

Code defaults are already **`auto`**: if artifacts exist and checks pass, they are used; otherwise **Numba** (CPU) or **MPS** (GPU).

### Live development flow (native-first)

1. Run `build-natives` + `start` (below).
2. Runs that use the native stack use **`kernel=cpu-sieve`** (worker **cpu**) and **`kernel=gpu-sieve`** (worker **gpu**) — continuous autopilot already enqueues these.
3. Check **`GET …/api/workers/native-stack`**: `resolved_backend: "native"` on CPU and Metal `available` / `would_use_metal` on GPU.

**Honesty:** this confirms you are **running** the native backends, not that global mathematical correctness is proven — see [`CORRECTNESS_AND_VALIDATION.md`](./CORRECTNESS_AND_VALIDATION.md). We keep **fallback** (Numba / MPS) when an artifact is missing so local dev does not break.

## 1. One-shot setup (recommended)

From the repository root:

```bash
brew install libomp   # OpenMP for .dylib (once)
bash scripts/mac-dev-stack.sh build-natives
```

Then start the stack as usual:

```bash
bash scripts/mac-dev-stack.sh start
```

Optional: rebuild on every **start** (slower on cold boot):

```bash
export COLLATZ_MAC_STACK_BUILD_NATIVES=1
bash scripts/mac-dev-stack.sh start
```

## 2. What `mac-dev-stack.sh` exports for workers

- `COLLATZ_CPU_SIEVE_BACKEND=auto` — loads `.dylib` when found.
- `COLLATZ_GPU_SIEVE_BACKEND=auto` — tries Metal (`metal_sieve_chunk` + successful ping), then MPS.
- `OMP_NUM_THREADS` aligned with `NUMBA_NUM_THREADS` if you did not set it (OpenMP for native `cpu-sieve`).

You can force: `COLLATZ_CPU_SIEVE_BACKEND=numba|native`, `COLLATZ_GPU_SIEVE_BACKEND=mps|metal` (strict modes documented in the backend docs).

## 3. Quick check (API)

With the API running:

- `GET /api/workers/native-stack` — combined JSON: `cpu_sieve_native` + `gpu_sieve_metal`.
- Or separately: `/api/workers/cpu-sieve-native`, `/api/workers/gpu-sieve-metal`.

Look for:

- `resolved_backend`: `"native"` for CPU when the `.dylib` is in use.
- `openmp_linked`: `true` if the library was built with OpenMP.
- `available` + `ping.ok` for Metal; `would_use_metal` when `auto`/`metal` would use the helper.

## 4. Without PyTorch (Metal-only `gpu-sieve`)

On headless machines without a useful MPS stack: `COLLATZ_GPU_SIEVE_METAL_WITHOUT_TORCH=1` plus a built `metal_sieve_chunk` — see [`GPU_SIEVE_METAL_AND_LIMITS.md`](./GPU_SIEVE_METAL_AND_LIMITS.md).

## 5. Related docs

- Correctness / validation protocol: [`CORRECTNESS_AND_VALIDATION.md`](./CORRECTNESS_AND_VALIDATION.md)
- CPU: [`CPU_SIEVE_NATIVE_BACKEND.md`](./CPU_SIEVE_NATIVE_BACKEND.md)
- GPU Metal: [`GPU_SIEVE_METAL_AND_LIMITS.md`](./GPU_SIEVE_METAL_AND_LIMITS.md)
- Throughput / tuning: [`PERFORMANCE_MACOS.md`](./PERFORMANCE_MACOS.md)
