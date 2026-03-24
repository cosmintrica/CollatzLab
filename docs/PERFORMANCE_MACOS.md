# Throughput: macOS vs PC (CPU sieve vs GPU sieve)

## Apples-to-apples

| What you run | Typical stack | Why it is fast |
|--------------|---------------|----------------|
| **`cpu-sieve`** on Ryzen 5 7600X | Numba `@jit(parallel=True)` + `prange`, or native `.dylib` + OpenMP | **Every odd seed** is scheduled across **all CPU cores**; checkpoint batches are **hundreds of millions** of odds (`COLLATZ_CPU_SIEVE_BATCH_SIZE`). |
| **`gpu-sieve` on NVIDIA + CUDA** | Custom Numba CUDA kernel | Massive parallelism with `seeds_per_thread` and few host round-trips. |
| **`gpu-sieve` on Apple Silicon** | Native **Metal** helper (auto) **or** PyTorch **MPS** | Same odd-descent contract as `cpu-sieve`. **Metal** when `metal_sieve_chunk` is built; else **MPS** (vectorized tensors — historically slow if sub-batches were tiny, see below). |

So **~250M/s on a 7600X** is expected for **`cpu-sieve`**, not a fair comparison to **`gpu-sieve` on MPS** unless you also measure **`cpu-sieve` on the Mac** (often tens–hundreds of M/s on M1 Pro-class silicon).

**Compute profile on macOS:** fresh SQLite seeds set the **GPU lane to 0%** by default so autopilot does not enqueue GPU work until you raise it (unified memory + UI contention). The dashboard compute panel shows a short **macOS GPU hint** when the API host reports Darwin and a GPU with kernels. Existing databases keep their saved profile until you change it in the UI.

**Native C `cpu-sieve` on macOS (optional, max throughput path):** `brew install libomp`, then `bash scripts/native_sieve_kit/build_native_cpu_sieve_lib.sh`. With `COLLATZ_CPU_SIEVE_BACKEND=auto` (default), the worker loads `libsieve_descent_native.dylib` when present. OpenMP uses **`OMP_NUM_THREADS`** and **`schedule(dynamic, 4096)`** on the per-seed loop so long-orbit seeds don’t leave other threads idle. `scripts/mac-dev-stack.sh` sets `OMP_NUM_THREADS` to match `NUMBA_NUM_THREADS` when you have not set `OMP_NUM_THREADS` yourself. Compare vs Numba on the same range: `python3 scripts/native_sieve_kit/benchmark_cpu_sieve_numba_vs_native.py`. Details: [`CPU_SIEVE_NATIVE_BACKEND.md`](./CPU_SIEVE_NATIVE_BACKEND.md).

### Why `cpu-sieve` throughput drops (e.g. ~400M/s → ~250M/s) on the same PC

The reported rate (M odd/s) depends on **checkpoint batch size** and **how many Numba threads** you use. Check in order:

1. **Compute profile in SQLite (dashboard)** — `cpu-sieve` scales the batch with the **CPU lane percentage** from the profile:  
   `batch ≈ COLLATZ_CPU_SIEVE_BATCH_SIZE × max(cpu_percent, 5) / 100` (see `_effective_checkpoint_interval` in `services.py`).  
   If the CPU slider is at **~62%**, the batch shrinks proportionally and you will see **fewer M/s** even when “nothing else changed.”
2. **Multiple CPU workers** (`COLLATZ_STACK_CPU_WORKERS` / macOS stack) — each worker gets **`NUMBA_NUM_THREADS` divided** by the worker count; a single run may use fewer cores → lower **per-run** throughput (aggregate across workers can still be similar).
3. **Environment variables** — `mac-dev-stack.sh` exports `COLLATZ_CPU_SIEVE_BATCH_SIZE=250000000`; if you start the worker **without** those exports, the code uses the in-code default (**250M** after recent alignment; it used to be **200M**).
4. **Interval / position in n** — near large values, descent steps are longer → fewer odds finished per second; always compare the same `[start,end]`.

**How to maximise throughput for a single run:** profile 100% system + 100% CPU lane, one CPU worker, raise `COLLATZ_CPU_SIEVE_BATCH_SIZE` (e.g. 400M–500M if you have RAM), `NUMBA_NUM_THREADS` = logical core count. If you use **native OpenMP**, set `OMP_NUM_THREADS` the same (or let `mac-dev-stack.sh` align them).

**Native C vs Numba** (methodology + script): [`CPU_NATIVE_VS_NUMBA.md`](./CPU_NATIVE_VS_NUMBA.md).

## Native Metal `gpu-sieve` (macOS modular path)

- **Helper binary:** `scripts/native_sieve_kit/metal/metal_sieve_chunk` — build: `bash scripts/native_sieve_kit/metal/build_metal_sieve_chunk.sh`.
- **Selection:** `COLLATZ_GPU_SIEVE_BACKEND` = `auto` (default), `mps`, or `metal`.
  - **`auto`:** use Metal when the helper exists and `--ping` succeeds; else MPS. If Metal throws, **fall back to MPS**.
  - **`metal`:** require Metal; no silent fallback.
  - **`mps`:** never use Metal (even if the binary exists).
- **Binary path:** `COLLATZ_METAL_SIEVE_BINARY` overrides; otherwise `COLLATZ_LAB_ROOT/scripts/native_sieve_kit/metal/metal_sieve_chunk` and a path derived from the installed package layout.
- **Chunk size (throughput-first auto):** If **unset** on **macOS** and **`COLLATZ_METAL_SIEVE_CHUNK_AUTO=1`** (default), order is: (**1**) **fresh calibration** from `data/metal_sieve_chunk_calibration.json` — write with `scripts/profile_metal_sieve_chunk.py --quick --reps 5 --write-calibration`; (**2**) else **RAM/swap ladder** (**16 M → 512 K**). Calibration is always **clamped** by the same RAM ceiling as the ladder. TTL default **30** days: `COLLATZ_METAL_SIEVE_CALIBRATION_MAX_AGE_DAYS`; disable file with `COLLATZ_METAL_SIEVE_USE_CALIBRATION=0`. Explicit `COLLATZ_METAL_SIEVE_CHUNK_SIZE` always wins. Cap: `COLLATZ_METAL_SIEVE_CHUNK_MAX` (default **16 777 216**; hard ceiling **67 108 864**). See **[`GPU_SIEVE_THROUGHPUT_STABILITY.md`](./GPU_SIEVE_THROUGHPUT_STABILITY.md)** for why M/s drops along a long run. Diagnostics include `calibration_*` keys on `GET /api/workers/gpu-sieve-metal` / `native-stack`.
- **Stdio overlap:** `COLLATZ_METAL_SIEVE_STDIO_PIPELINE` — default **`1`**: a reader thread buffers the next JSON line while the GPU runs the current chunk (`metal_sieve_chunk`). Set to **`0`** to force the legacy single-threaded stdio loop (debug / A/B).
- **Sweep script:** `PYTHONPATH=backend/src ./.venv/bin/python scripts/profile_metal_sieve_chunk.py --quick` (macOS + built helper). Add `--pipeline-ab` to compare pipeline on/off at the winning chunk size.
- **API:** `GET /api/workers/gpu-sieve-metal` returns diagnostics (safe on Linux/Windows — metadata only).
- **Windows / Linux:** no Metal code is loaded for `gpu-sieve`; CUDA and CPU paths unchanged.

Details and limits: [`GPU_SIEVE_METAL_AND_LIMITS.md`](./GPU_SIEVE_METAL_AND_LIMITS.md).

## Fix applied in-repo (MPS `gpu-sieve`)

Each checkpoint for `gpu-sieve` can cover **~500M linear values**. Inside MPS, work is split into configurable odd sub-chunks (`COLLATZ_MPS_SIEVE_BATCH_SIZE`, default **1M** — see *Evidence-based defaults* below).

### Streaming aggregates (critical)

Earlier builds materialised **full** NumPy arrays for every odd seed in the checkpoint (`total`, `stopping`, `max_excursion`) before `np.argmax` — e.g. **~250M × 16 bytes ≈ 4 GB per batch** plus GPU→host copies. That capped throughput around **sub‑1M/s** and felt “blocked”. The backend now **reduces max metrics chunk-by-chunk** with **O(chunk)** host memory only (same numerical results as before).

### Historical note (sub-chunk size)

Very small sub-chunks (**131072** odds) caused thousands of Python iterations + copies per checkpoint.

### Evidence-based defaults (Apple MPS)

`scripts/profile_mps_metal_sieve.py --quick` (4 combos on `[1, 800000]`) showed **lowest median wall time** for **`COLLATZ_MPS_SIEVE_BATCH_SIZE=1048576`** with **`COLLATZ_MPS_SYNC_EVERY=128`**. The same sweep had **worse** times for **`sync=256`** on both 1M and 2M batches — so defaults are **1M / 128** in code and `mac-dev-stack.sh`, and **`COLLATZ_MAC_EXTREME_THROUGHPUT=1`** bumps chunk to **2M** but **keeps sync 128** (not 4M/256).

Re-run the profile after major PyTorch/macOS upgrades; optimal knobs can shift.

Tune further (measure first):

```bash
export COLLATZ_MPS_SIEVE_BATCH_SIZE=4194304
export COLLATZ_MPS_SYNC_EVERY=128
```

### Scripts (tune before changing defaults)

From repo root (use `.venv` on Mac so MPS is available):

```bash
PYTHONPATH=backend/src ./.venv/bin/python scripts/benchmark_mac_throughput.py
PYTHONPATH=backend/src ./.venv/bin/python scripts/benchmark_mac_throughput.py --presets all
PYTHONPATH=backend/src ./.venv/bin/python scripts/benchmark_mac_throughput.py --quick
# Harder interval (large n, lower M/s, closer to long runs):
PYTHONPATH=backend/src ./.venv/bin/python scripts/benchmark_mac_throughput.py --hard-start 54000002001 --linear-end 10000000
```

**Systematic MPS sweep** (median time over batch size × sync cadence):

```bash
PYTHONPATH=backend/src ./.venv/bin/python scripts/profile_mps_metal_sieve.py --quick
PYTHONPATH=backend/src ./.venv/bin/python scripts/profile_mps_metal_sieve.py
PYTHONPATH=backend/src ./.venv/bin/python scripts/profile_mps_metal_sieve.py --linear-width 8000000 --reps 2 --json
```

Use **`--quick`** first (4 combos, smaller interval). The full 16-combo grid can run **30–90+ minutes**; long IDE/agent runs may be **aborted** before completion — run the full sweep in **Terminal.app** if needed.

Pick the preset with the best **M odd/s** and **parity=OK**, then export those `COLLATZ_MPS_*` values for the worker, or start the stack with:

```bash
COLLATZ_MAC_EXTREME_THROUGHPUT=1 bash scripts/mac-dev-stack.sh restart
```

(`EXTREME` defaults to 4M / 256 when `COLLATZ_MPS_*` are unset.)

**Automated random probes:** workers may enqueue small randomized runs (`owner=collatz-random-probe`) when the queue would otherwise stay empty, after research snacks. Set `COLLATZ_RANDOM_PROBES=0` to disable.

## “Use 100% CPU and GPU at once”

- **One run = one device + one kernel.** A run tagged `hardware=gpu` with `gpu-sieve` does **not** also execute on CPU cores for that same interval (that would be a different algorithm / split-scheduler, not implemented today).
- The dev stack **does** run **two workers in parallel** (same as Windows `dev-stack.ps1 -WithWorker`): **`mac-managed-worker-cpu`** claims CPU runs, **`mac-managed-worker-gpu`** claims GPU runs, so **two different runs** can advance at the same time and saturate both.

**Legacy / single worker:** `COLLATZ_MAC_STACK_SINGLE_WORKER=1 bash scripts/mac-dev-stack.sh start` → one process with `--hardware auto` (only one run at a time).

**Extra workers:** You can still start additional CLI workers with unique `--name` values if you want more than two consumers on the same SQLite queue (advanced).

## Practical advice on Mac

- If **`cpu-sieve` is hundreds of times faster** than **`gpu-sieve`** on your machine (common on M-series), **there is little reason to keep GPU verification streaming** burning wall time: set the dashboard **GPU lane** to **0%** (or stop the GPU worker) and let **CPU** + **`cpu-sieve`** own verification. Keep the GPU worker only if you care about **`gpu-collatz-accelerated`** experiments or cross-checks.
- For **maximum verification throughput** on Apple Silicon, prefer **`cpu-sieve`**; compare to **`gpu-sieve`** only if you are tuning MPS or validating parity.
- **GPU % in Activity Monitor** on the Python worker is the right signal for MPS; the dashboard cannot show GPU % on macOS without root tools.

## Why “M/s” drops on a long `gpu-sieve` / `cpu-sieve` run (not always a bug)

- `metrics.processed` counts **odd seeds** verified so far. As **`n` increases** along the interval, typical Collatz **descent trajectories get longer**, so **each batch takes more wall time** even if the implementation is healthy.
- The dashboard used to divide by the **full linear range width** (including evens) for progress — that understated progress and distorted averages; it now uses the **odd-seed total** for sieve-like kernels and shows **recent vs average** throughput between polls.

## Scaling workers (macOS + Windows)

- Set **`COLLATZ_STACK_CPU_WORKERS`** and **`COLLATZ_STACK_GPU_WORKERS`** (integers ≥ 1, caps 16 / 8). Scripts split **`NUMBA_NUM_THREADS` ≈ ceil(logical_cpus / CPU workers)** so multiple CPU workers do not each grab 100% of cores by default.
- **Single-GPU Mac (MPS):** **`COLLATZ_STACK_GPU_WORKERS` > 1** is usually **not** helpful (two processes contend on one GPU). Use **multiple CPU workers** plus **one GPU worker** to chew **different queued runs** in parallel.

## See also

- **[`RUN_RECOVERY.md`](./RUN_RECOVERY.md)** — stuck `running` jobs, `run release`, **`--migrate-cpu-sieve`**, and why **“CPU on Metal”** is not a thing (use **`cpu-sieve` on CPU**).
- **`scripts/recover-stuck-run.sh`** — wrapper around `python -m collatz_lab.cli run release …`.
- **Native Metal spike** (`scripts/metal_native_spike/`) — GPU throughput experiment only; not the lab’s validated sieve.
