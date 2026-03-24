# Native sieve kit (CPU C + Metal GPU)

**Purpose:** same **odd-only descent** contract as lab **cpu-sieve** / **gpu-sieve** (see `backend/src/collatz_lab/services.py` — `_collatz_sieve_parallel_odd`), for:

- **parity** (Python reference ↔ Numba ↔ C ↔ Metal);
- future **integration** (`research/NATIVE_SIEVE_PORT.md`).

## Quick checks

```bash
# From CollatzLab root
PYTHONPATH=backend/src python3 scripts/native_sieve_kit/compare_native_cpu_vs_lab.py
```

Builds `sieve_descent` with `cc` if missing, then compares **C vs pure Python ref vs Numba**.

**Lab `cpu-sieve` nativ (shared lib):**

```bash
# macOS: full kit (CPU .dylib + metal_sieve_chunk) from repo root:
#   brew install libomp
#   bash scripts/mac-dev-stack.sh build-natives
# Or CPU only:
bash scripts/native_sieve_kit/build_native_cpu_sieve_lib.sh
# Worker default: auto — uses native when the library is found (no export required).
# Force: export COLLATZ_CPU_SIEVE_BACKEND=native
```

The worker then uses the same Python finalize as Numba (overflow patch + aggregates). Details: [`docs/CPU_SIEVE_NATIVE_BACKEND.md`](../../docs/CPU_SIEVE_NATIVE_BACKEND.md).

**Benchmark Numba vs resolved path (`auto`):**

```bash
python3 scripts/native_sieve_kit/benchmark_cpu_sieve_numba_vs_native.py --start 1 --end 500000
```

## Metal (macOS)

**Production `gpu-sieve` helper** (used by the lab when `COLLATZ_GPU_SIEVE_BACKEND` allows):

```bash
bash scripts/native_sieve_kit/metal/build_metal_sieve_chunk.sh
./scripts/native_sieve_kit/metal/metal_sieve_chunk --ping
# Ping loads CollatzLabSieve.metallib + pipeline and checks CollatzChunkPartial ABI; JSON includes "metal_abi_ok":true.
```

**Dev verify binary** (parity smoke; `metal_sieve_verify` is a symlink to the same binary):

```bash
bash scripts/native_sieve_kit/metal/build_metal_verify.sh
./scripts/native_sieve_kit/metal/metal_lab_sieve_verify --base 1 --count 5000
# or: ./scripts/native_sieve_kit/metal/metal_sieve_verify --base 1 --count 5000
```

JSON line should match `compare_native_cpu_vs_lab.py` output for the same range.

**CPU native vs Numba throughput:**

```bash
PYTHONPATH=backend/src python3 scripts/native_sieve_kit/benchmark_cpu_native_vs_numba.py --odd-count 500000
```

## Files

| Path | Role |
|------|------|
| `sieve_descent.c` | Native CPU loop + `verify` / `bench` modes |
| `build_c.sh` | `cc -O3` binary |
| `metal/CollatzLabSieve.metal` | Lab-aligned kernel |
| `metal/verify_main.swift` | Dispatch + host reduction → JSON |
| `metal/build_metal_verify.sh` | `metal` + `metallib` + `swiftc` → `metal_lab_sieve_verify` + `metal_sieve_verify` symlink |
| `metal/chunk_main.swift` | `metal_sieve_chunk` (`--ping` / `--stdio` / one-shot) |

## Tests (CI)

- `backend/tests/test_sieve_reference_vs_numba.py` — always (if Numba present).
- `backend/tests/test_native_sieve_c_aggregate.py` — skips without `cc`.
- `backend/tests/test_metal_lab_sieve_parity.py` — skips off macOS or if Metal build fails.
