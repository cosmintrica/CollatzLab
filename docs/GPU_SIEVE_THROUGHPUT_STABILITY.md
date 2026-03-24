# Why **M/s** drops over time (and how the metric stays meaningful)

## 1. **n** increases along the interval (most common)

For **cpu-sieve** / **gpu-sieve**, `processed` counts **odd seeds** verified. As the run advances, **n** gets larger. For Collatz, typical orbits become **longer** (more steps until descent), so **cost per seed** rises.

**Consequence:** the same code on the same hardware may show **fewer M/s** at the end of the interval than at the start. That is not necessarily an implementation regression — it is the meaning of throughput “per verified seed”.

**Stability:** compare performance on the **same sub-interval** `[a,b]` or use the chunk benchmark (`profile_metal_sieve_chunk.py`) on a fixed interval.

## 2. Thermals / power limits (MacBook)

After minutes of **CPU + GPU** together, the SoC may **thermally throttle**. GPU M/s can drop below CPU for a while, then settle.

**Stability:** improve cooling, use `powermetrics` / Activity Monitor Energy, or temporarily separate load (single worker) as a controlled test.

## 3. Memory and **swap**

Large Metal chunks + many apps → **swap**. Under memory pressure the whole system slows.

**Stability:** watch swap; lower `COLLATZ_METAL_SIEVE_CHUNK_MAX` or re-run calibration; **auto** clamps calibration to a ceiling derived from RAM.

## 4. Two different runs on screen

If **CPU** and **GPU** process **different intervals** (different positions in n), they are **not** comparable as “who is faster in the abstract” — compare only on the same `[start,end]` and same kernel.

## 5. Auto-tuning **throughput-first** (calibration)

After `scripts/profile_metal_sieve_chunk.py --write-calibration`, the worker uses the benchmark **winning chunk** (clipped for RAM safety), not only an in-memory scale. Re-run calibration after **macOS / PyTorch / Metal helper** upgrades.

See also [`PERFORMANCE_MACOS.md`](./PERFORMANCE_MACOS.md).
