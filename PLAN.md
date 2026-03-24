# CollatzLab improvement plan

Derived from architectural analysis + cross-AI review (Claude / Cursor / ChatGPT).  
Date: 2026-03-23

---

## Quick context

SoT contract and descent semantics are sound. Fragility comes from:
- **`services.py` god object** (3743 lines: JIT kernels + orchestration + validation + scheduling)
- **Validation asymmetry**: `cpu-sieve` windows validated Numba-vs-Numba; `cpu-parallel-odd` windows validated Python-bigint-vs-Numba — different rigor, undocumented
- **No test comparing 3+ paths at once** on the same interval

---

## Priorities in order

### 1. Level-3 differential test + asymmetry documentation (same PR)

**Why now:** safety net before any change in `services.py`.

- Parametrized test: `sieve_reference` (Python) / Numba `cpu-sieve` / native C (skip if .dylib missing) / GPU sieve (skip if unavailable)
- Small safe interval, no int64 overflow: e.g. `[3, 1001]` and `[1, 9999]`
- Compare: `max_total_stopping_time`, `max_stopping_time`, `max_excursion`, `processed`
- Document in `_compute_descent_odd_only_reference`: "window reference = Numba consistency check, NOT bigint SoT; per-seed record validation uses `metrics_descent_direct` (bigint)"
- Extra gap: if a run sets no new records, `_validate_record_seeds` does no bigint work → CI differential test covers that

### 2. Refactor `services.py`

**Why after the test:** refactoring without the net risks undetected regressions.

Split into at least 4 modules:
- `kernels.py` — Numba JIT (`@njit`, `@cuda.jit`), compute constants
- `orchestration.py` — `execute_run`, `_CheckpointWriter`, `process_next_queued_run`
- `validation.py` — `validate_run`, `_validate_full_replay`, `_validate_random_windows`, `_validate_record_seeds`, helpers
- `scheduling.py` (or keep in `worker.py`) — autopilot, snack runs, hypothesis

### 3. Explicit COMPLETED → VALIDATED protocol

Clear rules, not “last run on stack” heuristics:
- Prioritize small near-validated runs over large recent runs
- Gap: run with no new records → bigint window mandatory (or CI differential as proxy)
- Documented in `research/WORKER_QUEUE.md`

### 4. `metrics_sot.py` — real contract module

Inverse of current dependency:
- `metrics_sot.py` defines protocol / interface
- `services.py` (or `kernels.py` after refactor) implements
- No circular lazy imports

### 5. Rest — cost-aware backlog

LLM/Gemini, Reddit intake, hypothesis sandbox: not wrong, but they do not buy rigor right now.

---

## What we do **not** do

- Cosmetic rename `metrics_descent_direct` → `metrics_descent_exact` at call sites (no real abstraction = wasted work)
- No edits to `services.py` before the differential test exists

---

## Current status

- [x] Level-3 differential test + asymmetry doc (`backend/tests/test_differential_cross_backend.py`)
- [x] `services.py` refactor — extracted into 5 modules: `metrics_sot.py` (146 L) + `validation.py` (534 L) + `_profile_helpers.py` (33 L) + `scheduling.py` (~490 L) + `orchestration.py` (~370 L); `services.py` reduced to ~2151 lines (from 3743)
- [x] COMPLETED → VALIDATED protocol — `_try_auto_validate` sorts smallest-first, oldest tiebreak
- [x] `metrics_sot.py` — zero imports from package; `validation.py` no longer imports `services` at module level
