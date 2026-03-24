# Stuck runs (GPU / MPS) — recover without losing progress

Large **`gpu-sieve`** jobs commit a SQLite checkpoint **after each batch** (`checkpoint.next_value`, `metrics.processed`, max records, etc.). If the worker is **hung** (driver, one giant batch, or UI that looks frozen), you can put the run back on the queue **without discarding** that state.

## 1. Stop the stuck worker

Stop the process that is executing the run (managed stack `stop`, or kill the worker terminal). If you skip this, it might fight the DB when the run is queued again.

## 2. Release the run to `queued`

From the repo root, with `PYTHONPATH=backend/src` and your venv:

```bash
python -m collatz_lab.cli run release COL-XXXX
```

Optional message:

```bash
python -m collatz_lab.cli run release COL-XXXX --note "Hung MPS batch; requeue after kill -9."
```

This:

- sets any worker with `current_run_id = COL-XXXX` to **idle**;
- sets the run from **running → queued**;
- **does not** clear `checkpoint_json` or `metrics_json`.

The next worker claim resumes from `checkpoint["next_value"]`.

## 3. Optional: continue on CPU (`cpu-sieve`)

Same **odd-sieve semantics** as `gpu-sieve` (lab benchmarks check parity). To move the queued run off the GPU:

```bash
python -m collatz_lab.cli run release COL-XXXX --migrate-cpu-sieve
```

This releases (if still `running`) then sets **`kernel=cpu-sieve`** and **`hardware=cpu`**. Only a **CPU** worker will pick it up.

## 4. Smaller GPU batches (fewer “frozen UI” gaps)

Checkpoint frequency is driven in part by **`COLLATZ_GPU_SIEVE_BATCH_SIZE`** (see `services._effective_checkpoint_interval`). Lower values → more frequent checkpoints and smaller MPS chunks (more overhead, but progress updates more often). Inner MPS tiling uses **`COLLATZ_MPS_SIEVE_BATCH_SIZE`** / **`COLLATZ_MPS_SYNC_EVERY`** — see [`PERFORMANCE_MACOS.md`](./PERFORMANCE_MACOS.md).

## 5. Metal spike vs lab correctness

The **Swift/Metal spike** under `scripts/metal_native_spike/` is a **throughput experiment** (tight uint64 loop). It is **not** wired into the lab’s validation pipeline and is **not** byte-for-byte the same artifact as a completed **`gpu-sieve`** run. Lab correctness for production runs comes from **`cpu-sieve` / `gpu-sieve`** parity checks and optional **validate** flows — not from the spike.

## 6. Fair Metal benchmark

To compare Metal spike vs PyTorch MPS on an idle GPU, **stop other GPU workers** first so nothing competes for the device.

## 7. “CPU on Metal?” — what to use on macOS

**There is no supported path that runs `cpu-sieve` (or any CPU kernel) *through* the Metal API.** In Apple’s stack:

| Goal | Use in Collatz Lab | Executes on | Notes |
|------|---------------------|------------|--------|
| **Fast sieve when the GPU is healthy** | `gpu-sieve`, `hardware=gpu` | Apple GPU via **PyTorch MPS** (Metal under the hood) | Normal production path. |
| **Stuck / unreliable GPU, or you want stability** | `cpu-sieve`, `hardware=cpu` | **CPU** (Numba / threads) | After `run release … --migrate-cpu-sieve`. Same odd-sieve semantics as `gpu-sieve`. |
| **Raw “how fast could a tight Metal kernel go?”** | `scripts/metal_native_spike/` | Apple GPU, **Metal Shading Language** directly | Benchmark only; **not** the lab’s validated sieve pipeline. |

So: **the recommended escape hatch for a bad GPU job is CPU + `cpu-sieve`, not “CPU on Metal”.** Metal is for **GPU** compute shaders; the lab does not ship a Metal-backed **CPU** runtime.

## 8. One-liner helper script

From repo root (uses `.venv/bin/python` if present):

```bash
bash scripts/recover-stuck-run.sh COL-0022
bash scripts/recover-stuck-run.sh COL-0022 --migrate-cpu-sieve --note "GPU hang"
```
