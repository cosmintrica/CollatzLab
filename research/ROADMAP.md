# Collatz Lab Research Roadmap

## Mission

Build a local-first research lab that helps us:

- run reproducible Collatz experiments;
- compare multiple research directions in parallel;
- record what was tried, what failed, and what remains promising;
- connect claims and partial proofs to concrete evidence;
- prepare strong candidates for later formalization.

This project is not based on the assumption that brute force alone will solve the conjecture. Compute is used as a filter, validator, and pattern-mining tool.

## Phase 0: Foundation

Status: implemented in v1 baseline.

Goals:

- local repository with backend, dashboard, research notes, artifacts, and reports;
- stable IDs for tasks, runs, claims, and artifacts;
- reproducible run storage in SQLite;
- CPU execution and independent validation;
- local dashboard for visibility.

Deliverables:

- CLI for init, tasks, runs, claims, validation, reviews, and reports;
- FastAPI endpoints for runs, claims, directions, tasks, and artifacts;
- Markdown notes in `research/`;
- generated evidence in `artifacts/` and `reports/`.

## Phase 1: Research Operating System

Goals:

- make the lab understandable at a glance;
- support several active directions without dashboard overload;
- improve branch discipline and intermediate-proof storage;
- define promotion and abandonment rules clearly.

Work items:

1. Dashboard UX refinement
- split UI into tabs or views;
- add legends, short explanations, and show-more controls;
- keep summary visible while reducing default noise.

2. Research workflow hardening
- one branch per idea or experiment: `codex/<track>-<id>`;
- every claim links to supporting or refuting runs;
- every direction gets periodic review and score update.

3. Better artifacts
- richer run summaries;
- direction review reports;
- claim pages with stronger dependency tracking.

Acceptance:

- a new contributor can understand the state of the lab in a few minutes;
- evidence can be traced from dashboard to note/report/run;
- weak directions can be frozen without deleting history.

## Phase 2: Stronger Compute

Status: largely implemented.

Goals:

- scale interval experiments and replay them reliably;
- compare implementations and kernels;
- GPU and distributed execution.

Work items:

1. CPU improvements — **implemented**
- chunked execution with stronger checkpointing;
- 4 configurable kernels (see below);
- larger interval batches (250M CPU, 500M GPU) and record tracking.

2. GPU worker — **implemented** (NVIDIA CUDA); **portability — planned**
- separate managed worker process, not mixed into the API server;
- same run metadata shape as CPU runs;
- GPU kernel with CUDA shared-memory block reduction;
- int64 overflow guard with automatic bigint fallback to cpu-parallel;
- replay on small ranges against CPU before trusting larger sweeps.
- **Platforms today:** primarily NVIDIA CUDA on Windows/Linux x86_64 where Numba sees a driver.
- **Roadmap (gradual):** see [`research/HARDWARE_AND_KERNELS.md`](./HARDWARE_AND_KERNELS.md) — Apple Silicon (Metal family), Windows on ARM, Linux ARM64, AMD ROCm, Intel GPU stacks, integrated vs dedicated detection; CPU kernels remain the universal baseline.

### Kernel reference

| Kernel | Description | Typical throughput |
|--------|-------------|-------------------|
| `cpu-direct` | Single-threaded Python, per-seed `metrics_direct()` | ~50K val/sec |
| `cpu-parallel` | Numba JIT, parallel across all CPU cores | ~7M val/sec |
| `cpu-parallel-odd` | Like cpu-parallel but iterates only odd seeds (2x throughput, mathematically equivalent via v2 induction) | ~14M val/sec |
| `gpu-collatz-accelerated` | CUDA kernel with per-block shared-memory reduction, 256 threads/block | ~20-50M val/sec |

### Overflow recovery

When the GPU kernel detects a seed whose trajectory exceeds int64 limits, it triggers an overflow guard. The system automatically:
1. Marks the GPU run as FAILED with a diagnostic summary.
2. Creates a `recover-prefix` run that preserves the already-computed safe portion as COMPLETED.
3. Creates a `recover-tail` run using `cpu-parallel` with Python bigint arithmetic for the remaining range.
4. Deduplicates recovery runs so restarts do not create redundant work.

3. Distributed execution
- optional remote workers later;
- queue and lease model for parallel agents or machines;
- reproducible environment snapshots.
- central coordinator, worker registration, shard assignment, artifact upload, dedupe, and independent revalidation.

Target operating model for this phase:

- keep the project open source;
- run one main platform under the maintainer's control as the canonical coordinator;
- let outside contributors run a separate `worker-agent` on their own machines;
- send contributed runs, artifacts, and validation payloads back to the maintainer-controlled database;
- never trust remote compute blindly; important results must still be revalidated on independent hardware before promotion.

Acceptance:

- CPU and GPU agree on controlled validation ranges;
- interrupted runs resume safely;
- larger batches can be scheduled without manual bookkeeping.

## Phase 3: Theory Workspace

Goals:

- turn raw observations into structured mathematical work;
- separate ideas, promising claims, supported claims, and refutations;
- reduce the chance of repeatedly exploring the same dead end.

Active directions:

1. Verification
- interval sweeps;
- stopping-time and excursion records;
- implementation comparisons.

2. Inverse-tree parity
- reverse tree exploration on odd nodes;
- residue-class filters;
- parity-vector heuristics.

3. Lemma workspace
- candidate lemmas;
- implication chains;
- counterexamples and invalidations;
- intermediate proof drafts.

Promotion rules:

- a direction becomes `promising` only after reproducible evidence from multiple runs or an actual search-space reduction;
- a claim becomes `supported` only when linked evidence exists;
- a direction becomes `refuted` when a direct contradiction or reliable failed reproduction appears;
- weak work becomes `frozen`, never deleted.

## Phase 4: Formalization

Goals:

- move only mature results into a proof assistant;
- avoid formalizing unstable ideas too early.

Planned approach:

- keep v1 and v2 mostly in Markdown plus reproducible compute;
- introduce Lean only for claims marked `formalize`;
- formalize the strongest invariant candidates, not the entire notebook.

Acceptance:

- a formalization candidate has a stable statement, dependencies, evidence links, and no unresolved reproduction issues.

## Phase 5: Remote Hosting and Sharing

Not needed now, but planned.

Goals:

- keep the frontend compatible with Vite development and future Vercel hosting;
- separate frontend hosting concerns from backend orchestration.

Planned approach:

- keep the dashboard as a Vite app;
- use `VITE_API_BASE_URL` so the frontend can point to a local API now and a hosted API later;
- for local managed stack, serve a stable built snapshot;
- for development, continue supporting Vite hot reload;
- when needed, deploy the dashboard separately to Vercel.
- keep the backend ready for a later coordinator mode so multiple outside workers can centralize results into one lab.

Longer-term hosted shape:

- the public website can live on Vercel as the main dashboard surface;
- the canonical coordinator API and database stay under the maintainer's control;
- compute does not happen on Vercel; it happens on local or remote worker agents.

## Phase 6: Federated Open Compute

Planned operating model:

- one maintainer-owned main platform acts as the canonical source of truth;
- the repo remains open source;
- anyone can install and run a `worker-agent` locally to contribute bounded compute;
- worker agents register capabilities, ask for jobs, execute them locally, and upload results;
- the central platform deduplicates work, tracks provenance, and revalidates important results.

Core requirements:

- worker registration and capability discovery;
- signed or authenticated job/result exchange;
- shard assignment for bounded intervals or bounded theory probes;
- artifact upload plus reproducibility metadata;
- trust and quarantine rules for unverified remote results;
- independent rechecks before a remote result affects claim status.

## Next: Hypothesis Math Upgrades (urgent)

Priority improvements for `hypothesis.py` — do soon, before scaling the battery to larger ranges.

### 1. Stratified residue analysis (fix confounding)

**Problem:** `analyze_residue_classes` compares mean TST across classes globally but does not control for seed magnitude. Larger seeds naturally have higher TST (≈ 6.95 · log₂n). A class that happens to contain more large seeds looks anomalously slow — classic Simpson's paradox.

**Fix:** Bin seeds by log₂n (like `test_stopping_time_growth` already does), then compute residue class deviations *within* each bin. A class is anomalous only if it deviates consistently across multiple magnitude bins, not just in aggregate.

**Deliverable:** New or upgraded `analyze_residue_classes_stratified()` that returns per-bin z-scores and an overall verdict.

### 2. Glide / odd-step ratio analysis (new probe)

**Problem:** The current battery tracks total stopping time and max excursion but ignores the *internal structure* of orbits — specifically the ratio of odd steps to even steps (glide). This ratio is directly linked to 2-adic descent behavior and is the core mechanism behind why orbits shrink.

**Fix:** Extend `metrics_direct` to also return `odd_steps` count. Add a new hypothesis function that studies odd-step fraction per seed, per residue class mod 2^k, and per magnitude bin. Compare against the Terras heuristic (expected odd fraction ≈ log₂(3)⁻¹ ≈ 0.63).

**Deliverable:** New `analyze_glide_structure()` function in hypothesis.py, included in the battery.

### 3. Battery scalability test (meta-analysis)

**Problem:** All hypotheses are currently tested at a single range (default 50k). We don't know if "plausible" results survive at 200k, 1M, or 10M. A pattern that appears at 50k but vanishes at 500k is a false signal.

**Fix:** Run the battery at escalating ranges (50k → 200k → 1M) and compare status stability. Flag any hypothesis whose status flips between ranges. This is itself a hypothesis about hypothesis robustness.

**Deliverable:** New `test_battery_scalability()` orchestrator or a CLI command that runs the battery at multiple scales and produces a stability report.

---

## Rules of Work

- never trust a pattern from one run;
- never promote a direction without evidence;
- never delete failed ideas if they teach us what not to repeat;
- every important result must be reproducible from saved metadata;
- every serious branch should end in one of: report, claim, refutation, or frozen note.

## Current Open Questions

- how much of the theory workspace should be template-driven;
- when to add a queue for multiple machines instead of one local node;
- whether GPU utilization (~20% due to warp divergence) justifies further kernel optimization or should be accepted as ceiling for this workload.
