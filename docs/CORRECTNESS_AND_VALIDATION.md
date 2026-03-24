# Correctness and validation — practical protocol (Collatz Lab)

This document is **platform-wide**: it applies to Collatz Lab on **macOS, Linux, and Windows**, and to **every** fast backend we ship (Numba, native C / OpenMP, Metal, PyTorch MPS, CUDA). macOS-only notes (e.g. Metal) are one deployment flavor, not the definition of source-of-truth.

Code entrypoints for the contract are centralized in `collatz_lab.validation_source` (metadata + `metrics_descent_exact`); see also `GET /api/validation/contract`.

This document connects **general guidance** (overflow, integer types, step definitions) to **what the repository does today** and sensible next steps. It does not replace a mathematical proof; it describes **engineering confidence** in computational results.

## In short: can we say “100% sure”?

**No** in the sense of “proven for every seed in the universe.” **Yes** in the sense that, for the **implemented kernel contract** (e.g. odd-only descent for `cpu-sieve` / `gpu-sieve`), we have:

- explicit overflow semantics (signalling, not silent wraparound);
- the same Python **finalize** (aggregates + bigint patch) on aligned fast paths;
- automated parity tests on intervals and targeted cases.

Confidence grows through **cross-validation**, not by declaring one backend “the truth.”

---

## 1. Source of truth (SoT)

**Ideal (research engineering):** a **Python + arbitrary-precision integer (bigint)** backend, sequential, without shortcuts, that for each seed produces full metrics and optionally a trajectory hash.

**In this repo (all platforms):**

- **`metrics_descent_direct`** (also exposed as **`validation_source.metrics_descent_exact`**) — exact-arithmetic descent metrics for overflow recovery and exact paths (used in the patch pipeline).
- **`sieve_reference`** (`sieve_reference.py`) — slow mirror of odd-only sieve logic; documented **not** to reproduce Numba’s int64 overflow patch alone; used for **parity on ranges without overflow** (`test_sieve_reference_vs_numba`).

**Conclusion:** the strict SoT for the whole kernel family is the **combination** of the Python reference (where applicable) + **bigint** where int64 is insufficient. Numba / native C / Metal / CUDA / MPS are **validated fast backends**, not absolute truth.

---

## 2. Backend contract (fast paths on every OS)

For lab sieve kernels, fast paths return compatible arrays/aggregates; when int64 limits are exceeded:

- **no silent wraparound** treated as valid values;
- problematic seeds are **marked** (e.g. `-1` in tables) and routed through **`metrics_descent_direct`** / `_OverflowPatch` as in `services.py`.

The generic protocol’s **minimal** output (total_steps, max_value, overflow, …) **maps** onto existing structures (`AggregateMetrics`, validation payloads); duplicating the same JSON at every layer is unnecessary if the validation checksum contract stays explicit.

---

## 3. Overflow — formal pipeline

1. The fast backend runs until normal termination or until the kernel hits an **overflow boundary**.
2. Overflow seeds are **not** continued in int64 by the fast backend; they are **patched** in Python with bigint.
3. Final aggregates reflect those patches (same reduction code on aligned paths).

This is already the **-1 + Python finalize** line; treat it as a **contract**, not an implementation detail.

---

## 4. Test levels (ideal requirements vs repo today)

| Level | Goal | In repo (examples) |
|-------|------|-------------------|
| **1** — small deterministic set | bit-perfect aggregate metrics / small seeds | `test_sieve_reference_vs_numba`, `test_cpu_sieve_native_parity`, overflow tests |
| **2** — known hard cases | 27, 32/64-bit boundaries, large excursions | partly in kernel/overflow tests; can be extended explicitly |
| **3** — differential batch | same interval: reference vs Numba vs native vs Metal | native–Numba parity; Metal vs MPS where tests exist; **not** yet one suite for “all backends on the same random batch” |
| **4** — fuzz + replay artifact | on mismatch: seed, step, values, repro command | partly via artifacts/runs; a **dedicated DB schema for divergences** is future work |

---

## 5. DB schema (protocol vs reality)

The protocol suggests `validation_artifacts`, `validation_level`, etc. **Collatz Lab** already centres run validation on **runs + checksum / status / validate**, without a separate subsystem for every protocol field.

**Recommendation:** stay **minimal** until the first systematic mismatch appears; then add a **JSON artifact** (or table) with `seed`, `first_divergence_step`, `backend_a` / `backend_b`, `repro_command`.

---

## 6. When to mark a run “validated”

Per project philosophy (see README): **validated** means **replay / independent path**, not “ran without crashing.”

- Not: “processed 10⁹ numbers, looks fine.”
- Yes: **bit-perfect** match on the agreed validation payload, or explicit equivalence after overflow handling.

---

## 7. Red flags

- High volume without **differential** checks against SoT or another backend.
- “Numba is truth” or “Metal is truth.”
- Ignoring overflow because it is “rare.”
- Comparing different metrics (step / stopping-time definitions) across kernels without an explicit mapping.

---

## 8. What already works well in this project

- Separation of **runs** vs **claims** and a conservative stance on external sources.
- Centralised **bigint patch** for overflow.
- **Shared finalize** for native and Numba `cpu-sieve`.
- Modular Metal integration with documented fallback where applicable.
- Parity tests for critical paths.

---

## 9. Where the “ideal” protocol should be **adapted** (and that is OK)

1. **`trajectory_hash` for billions of seeds** — prohibitively expensive if computed per seed on every run. At large scale: **aggregates + sampling** of full trajectories on subsets, or hashing only in “debug / research” mode.
2. The **same odd-only contract** on `cpu-sieve` / `gpu-sieve` reduces “different step definition” risk **between those kernels**, but does not automatically align them with a naive “one step = n/2 or 3n+1” kernel without acceleration — label clearly what each kernel measures.
3. **Level 4 fuzz** — fits **nightly** or manual jobs, not necessarily every PR (CI time).

---

## 10. “Past 2^71” — does it still mean anything?

- **Community / literature framing:** phrases like “verified up to 2^71” refer to **distributed or published computational campaigns** that checked every integer in `[1, N]` under an agreed step definition. That is **stronger empirical evidence**, not a proof of the Collatz conjecture.
- **Inside this lab:** bulk kernels use **signed int64** fast paths with explicit overflow handling (`COLLATZ_INT64_ODD_STEP_LIMIT` and related guards in `metrics_sot.py` / sieve backends). Seeds that would overflow int64 on the fast path are **not** silently wrapped; they are routed through **`metrics_descent_direct`** (Python **bigint**) in the finalize / patch pipeline — see §2–§3 above.
- **SoT reference:** `metrics_direct` and `metrics_descent_direct` use **arbitrary-precision** Python integers; there is **no** hard `2^71` ceiling in those functions. The frontier matters for **what has been independently checked at scale**, not for whether bigint repair is “allowed” to run.
- **Practical takeaway:** extending your own verified interval still has **engineering value** (regressions, parity across kernels). It does **not** close the conjecture mathematically.

---

## See also

- API: `GET /api/validation/contract` — JSON summary of SoT scope and entrypoints.
- Native stack on macOS (one deployment): [`MACOS_FULL_NATIVE_CPU_AND_METAL_GPU.md`](./MACOS_FULL_NATIVE_CPU_AND_METAL_GPU.md)
- Native CPU: [`CPU_SIEVE_NATIVE_BACKEND.md`](./CPU_SIEVE_NATIVE_BACKEND.md)
- Metal: [`GPU_SIEVE_METAL_AND_LIMITS.md`](./GPU_SIEVE_METAL_AND_LIMITS.md)
