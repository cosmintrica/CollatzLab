# Development Backlog

This backlog is intended for autonomous or semi-autonomous development runs.

Rules:

- prefer real data and real workflows over placeholders;
- do not introduce mock data unless explicitly requested;
- keep the frontend compatible with Vite and future Vercel deployment;
- run relevant tests or builds after each bounded change;
- update this file when a priority is completed or replaced.

## Current Priorities

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
- expose direction review outputs more clearly.

5. Hosting readiness
- keep frontend build clean for Vercel later;
- keep local managed stack stable;
- preserve a separate Vite mode for development.

## Definition of Done for One Autonomous Run

- complete one bounded improvement;
- run the smallest relevant verification step;
- update docs or backlog if priorities changed;
- produce a short summary of what changed, what remains, and any risk.
