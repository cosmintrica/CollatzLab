# Native Metal spike (performance probe only)

This is **not** production Collatz Lab code. It answers: *roughly how fast could a native Metal compute path process many odd seeds on this Mac?*

## What it does

- Compiles `CollatzSpike.metal` to a `.metallib` via Xcode command-line tools.
- Runs a Swift driver that dispatches **one thread per odd seed** in `[base, base+2*(count-1)]`.
- Each thread runs an **odd-only descent** loop (stop when `n < seed`, same idea as `cpu-sieve` / MPS sieve) with uint64 overflow guard.
- Prints **wall time** and **approximate odd seeds / second**.

For **stable** `odd_per_sec`, prefer a larger `--count` (e.g. `5000000`): when `wall_s` is only a few milliseconds, timer noise dominates.

## Compare to PyTorch MPS

After a successful spike run, compare to the lab’s Python path on a similar odd count (same machine, close `base`):

```bash
cd /path/to/CollatzLab
PYTHONPATH=backend/src ./.venv/bin/python -c "
from time import perf_counter
from collatz_lab.services import compute_range_metrics
from collatz_lab.hardware import GPU_SIEVE_KERNEL
base, span = 1, 200_001  # adjust to match spike count of odds
t0 = perf_counter()
compute_range_metrics(base, base + span - 1, kernel=GPU_SIEVE_KERNEL)
print('MPS wall_s', round(perf_counter()-t0, 3))
"
```

If Metal spike is **orders of magnitude** faster than MPS for the same `N`, a native kernel is worth engineering; if it is **similar or slower**, the bottleneck may be memory / sync / Python elsewhere, not “missing Metal”.

**Fairness:** the spike measures **raw Metal kernel throughput** (steps written to a buffer, minimal host logic). Lab **`gpu-sieve`** times include **PyTorch**, **batching**, **`.any()`/sync cadence**, and **aggregate metrics** back to Python — so `compare_with_lab.py` often shows **very large ratios** (e.g. 10³–10⁴× on Apple Silicon). That means “the GPU can run a tight loop this fast” vs “the current product path costs this much”; it does **not** mean the lab implementation is wrong. Use the ratio as an **upper bound** on how much a highly optimized native path could win if you moved more of the pipeline to Metal.

## Requirements

- macOS with **Xcode** or **Command Line Tools** plus the **Metal Toolchain** component (needed for `xcrun metal`).
- Apple Silicon (or AMD Metal GPU) — Intel Macs without Metal GPU may fail.

If `metal` fails with *missing Metal Toolchain*:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## Run

From the repo root:

```bash
bash scripts/metal_native_spike/run.sh --count 500000 --base 1
```

Or from this directory:

```bash
bash run.sh --count 500000 --base 1
```

## One-shot comparison vs lab `gpu-sieve` (MPS)

Same odd workload (default 500k odds from base 1), then prints **odd/s** for Metal vs PyTorch MPS:

```bash
cd /path/to/CollatzLab
PYTHONPATH=backend/src python3 scripts/metal_native_spike/compare_with_lab.py
```

With venv:

```bash
./.venv/bin/python scripts/metal_native_spike/compare_with_lab.py --preset-env tuned
```

Options: `--count`, `--base`, `--preset-env default|tuned|extreme`, `--no-warmup`.

> **CI / cloud agents** often lack `metal` and MPS; run this **on your Mac** with Xcode Metal toolchain + project venv.
