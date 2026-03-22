# Development Backlog

This backlog is intended for autonomous or semi-autonomous development runs.

Rules:

- prefer real data and real workflows over placeholders;
- do not introduce mock data unless explicitly requested;
- keep the frontend compatible with Vite and future Vercel deployment;
- run relevant tests or builds after each bounded change;
- update this file when a priority is completed or replaced.

## Current Priorities

0. Immediate next after hosting pass
- keep continuous chained runs alive longer so `LIVE NOW` stays visible more often;
- refactor `hardware.py` into platform adapters for Windows/Linux/macOS and x86_64/ARM64;
- sketch a distributed coordinator backend so remote workers can contribute compute and centralize evidence.

1. Worker queue and hardware-aware execution
- define a real queue/lease model for runs;
- keep CPU as the default worker path and allow GPU dispatch only when the run contract supports it;
- preserve checkpoint/resume semantics across worker types;
- document the execution contract in `research/WORKER_QUEUE.md`.

2. Dashboard UX
- keep the warmer visual language the user preferred;
- make navigation clearer with tabs or menu structure;
- add search, filtering, and detail drawers for runs, claims, and tasks;
- preserve good performance and avoid heavy visual effects.

3. Real data fidelity
- avoid fake or placeholder summary values;
- show loading, empty, and error states explicitly;
- ensure the UI always points to the real API base URL.

4. Research workflow
- add richer evidence inspection for runs and artifacts;
- dashboard quick actions for run queue, task creation, claim creation, and direction review are in place;
- next: link claims to runs from the UI and add deeper run/artifact inspection;
- source review now keeps structured history and Gemini can draft guarded review suggestions;
- Gemini autopilot now has guarded task planning and can create bounded tasks from local history;
- next: show autopilot status more clearly, keep a visible audit trail, and prevent duplicate low-value loops;
- expose direction review outputs more clearly.

5. Structure analysis beyond orbit playback
- add tools that inspect residue classes, predecessor branching, and clustering patterns, not only stopping time and excursion;
- surface completed-run analytics in a way that helps theory work, not just compute tracking;
- use inverse-tree and parity ideas as first-class analysis modules in the dashboard.

6. Indirect approaches
- keep a low-confidence workspace for surrogate maps, bijection attempts, and indirect formulations;
- treat these as theory tasks with explicit proof obligations, not as implicit progress toward a proof.

7. Hosting readiness
- keep frontend build clean for Vercel later;
- keep local managed stack stable;
- preserve a separate Vite mode for development.

8. Cross-platform execution and hardware detection
- split hardware discovery into backend adapters for Windows, Linux, and macOS;
- support CPU-first execution everywhere, with GPU backends added per platform rather than hardcoding NVIDIA/Windows assumptions;
- detect architecture and runtime backend explicitly: x86_64 vs ARM64, NVIDIA CUDA vs Apple Metal vs CPU-only;
- keep a portable fallback path so Mac or ARM users can still run real experiments even without GPU kernels.
- make local kernel selection automatic based on detected hardware class, runtime readiness, and validation safety.

9. Federated / distributed lab mode
- design a central coordinator that can assign bounded run shards to remote workers;
- register remote workers with capability metadata, trust level, and reproducibility rules;
- upload results, artifacts, and validation proofs back to a central registry;
- deduplicate overlapping work and revalidate important results on independent hardware before promotion.
- preserve an open-source `worker-agent` that anyone can run on their own machine;
- keep the main site, coordinator, and canonical database under maintainer control;
- treat contributed remote compute as provisional until replayed or independently validated.

10. Hosted main platform model
- keep the public dashboard hostable on Vercel or similar frontend hosting;
- do not depend on frontend hosting for compute or hardware access;
- separate roles clearly:
  - public dashboard
  - maintainer-owned coordinator API/database
  - contributor-run local worker agents

## Definition of Done for One Autonomous Run

- complete one bounded improvement;
- run the smallest relevant verification step;
- update docs or backlog if priorities changed;
- produce a short summary of what changed, what remains, and any risk.
