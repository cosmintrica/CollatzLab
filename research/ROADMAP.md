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

Goals:

- scale interval experiments and replay them reliably;
- compare implementations and kernels;
- prepare for GPU and distributed execution.

Work items:

1. CPU improvements
- chunked execution with stronger checkpointing;
- configurable kernels;
- larger interval batches and record tracking.

2. GPU worker
- separate worker contract, not mixed into the API server;
- same run metadata shape as CPU runs;
- replay on small ranges against CPU before trusting larger sweeps.

3. Distributed execution
- optional remote workers later;
- queue and lease model for parallel agents or machines;
- reproducible environment snapshots.

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

## Rules of Work

- never trust a pattern from one run;
- never promote a direction without evidence;
- never delete failed ideas if they teach us what not to repeat;
- every important result must be reproducible from saved metadata;
- every serious branch should end in one of: report, claim, refutation, or frozen note.

## Current Open Questions

- when to introduce the GPU worker;
- how large the first meaningful validation intervals should be;
- how much of the theory workspace should be template-driven;
- when to add a queue for multiple machines instead of one local node.
