# Collatz Lab

Local-first research platform for the Collatz conjecture.

Collatz Lab is not a proof claim and not a brute-force toy. It is a workspace for running reproducible CPU/GPU experiments, validating compute results, tracking mathematical claims, reviewing external sources, and keeping theory and evidence separated.

## Why this exists

The Collatz conjecture is still open. Computational verification is useful, but it is not a proof. This project is built around that distinction.

The goal is to support several research activities in one place:

- queueing and replaying real Collatz runs;
- validating results through independent implementations;
- recording claims, links, artifacts, and reports;
- reviewing external proof attempts and tagging common failure modes;
- exploring non-brute-force directions alongside verification;
- visualizing live checkpoints, orbit math, and derived formulas from real runs.

## Current capabilities

**Backend**

- Python 3.13 package: orchestration, SQLite persistence, validation, reports, Reddit feed helpers, hypothesis utilities
- FastAPI API for the dashboard and local automation (`collatz-lab-api` / `uvicorn`)
- CLI (`python -m collatz_lab.cli` or `lab`) for tasks, runs, claims, workers, reports
- Resumable worker loop (CPU and GPU), checkpointed execution, optional validate-after-run
- **Compute profile** persisted in the DB: system / CPU / GPU lane percentages, continuous compute on/off; workers apply batch sizing, thread counts, GPU throttling, and idle pacing so the budget is visible in practice
- **Logging:** Numba/CUDA driver memory chatter is suppressed in workers; `/api/logs` aggregates `data/logs/worker-*.log` and filters legacy noise unless you search for it

**Dashboard** (React 19 + Vite 7)

- Tabbed workspace: overview, evidence, operations (compute, tracks, guide), live math, paper view, and related flows
- **Light / dark theme** (manual toggle, persisted); favicon and app mark switch with the theme
- **Compute budget** rail: start/stop continuous compute, sliders for whole-system / CPU / GPU lanes, apply to backend profile
- **Active runs** strip with progress, throughput hints, and checkpoint status
- **System logs** panel: search, level/source filters, refresh (worker log tail + failed-run summaries)
- **Run rail**, **Reddit intel rail** (r/Collatz intake, not auto-trusted), **live math** navigator / ticker / orbit UI, **research paper** page (KaTeX, `research/paper.json`)
- Configurable API base via `VITE_API_BASE_URL` for hosted frontends

**Data & research**

- SQLite under `data/` for runs, workers, claims, sources, compute profile, runtime settings
- `research/`: roadmap, claim markdown files, paper metadata, incident/LLM notes as applicable
- Generated outputs: `artifacts/`, `reports/` (gitignored by default for local runs)

**Tooling**

- GitHub Actions CI on `main`: backend `pytest`, dashboard `npm ci` + `npm run build`
- Root `package.json` scripts for the dev stack (PowerShell on Windows); `scripts/*.sh` for backend, dashboard, and worker on Unix-like systems

## Research stance

This repo treats the problem conservatively:

- `validated result` means an experiment replay matched through an independent path
- `validated result` does not mean a proof
- external sources are intake material until reviewed
- brute force is treated as evidence and falsification tooling, not the final strategy
- theory work lives in claims, directions, source reviews, and lemma testing
- Gemini can assist with review and planning, but it never becomes the source of truth

## Repository layout

- `backend/`: Python package, CLI, API, services, worker loop, tests
- `dashboard/`: Vite + React UI (`npm` lifecycle inside this folder)
- `research/`: roadmap, direction notes, claim documents, paper JSON
- `artifacts/`: generated JSON, Markdown, and evidence outputs
- `reports/`: generated lab reports
- `data/`: SQLite database and runtime state (local)
- `scripts/`: dev stack (PowerShell), plus optional shell helpers for API / Vite / workers

## Quickstart

1. **Python:** 3.13+ recommended (matches `pyproject.toml` and CI).

2. Install backend dependencies:

   ```powershell
   pip install -e .\backend[dev]
   ```

   Optional CUDA wheels for GPU workers (environment-specific; see Numba/CUDA docs for your GPU):

   ```powershell
   pip install -e ".\backend[dev,gpu]"
   ```

3. Install dashboard dependencies:

   ```powershell
   npm install --prefix .\dashboard
   ```

4. Configure local environment if needed:

   - edit `.env`
   - or copy `.env.example` to `.env`
   - if no Gemini key is detected, the dashboard can prompt for it and store it in the local `.env`

   Example Gemini setup:

   ```text
   COLLATZ_LLM_ENABLED=1
   GEMINI_MODEL=gemini-2.5-flash
   GEMINI_API_KEY=your_key_here
   COLLATZ_LLM_AUTOPILOT_ENABLED=1
   COLLATZ_LLM_AUTOPILOT_INTERVAL_SECONDS=1800
   COLLATZ_LLM_AUTOPILOT_MAX_TASKS=3
   ```

5. Initialize the workspace:

   ```powershell
   python -m collatz_lab.cli init
   ```

6. Start the managed local stack (API + Vite + optional worker):

   ```powershell
   npm run stack:start:worker
   ```

7. Open:

   - dashboard: `http://localhost:5173`
   - API health: `http://127.0.0.1:8000/health`

## Vercel frontend hosting

The dashboard is prepared for a frontend-only Vercel deploy.

- `vercel.json` at the repo root points Vercel to `dashboard/`
- the frontend reads `VITE_API_BASE_URL`
- if `VITE_API_BASE_URL` is not set, the site falls back to the local API URL for desktop/local use

For a hosted frontend, set `VITE_API_BASE_URL` in Vercel to the public backend URL you want the dashboard to call.

## Local commands

Managed stack (Windows, from repo root):

```powershell
npm run stack:start
npm run stack:start:worker
npm run stack:start:cpu
npm run stack:start:gpu
npm run stack:start:vite
npm run stack:start:vite:worker
npm run stack:status
npm run stack:stop
npm run stack:restart
```

Other root scripts:

```powershell
npm run dashboard:dev
npm run dashboard:build
npm run backend:test
```

CLI examples:

Create a task:

```powershell
python -m collatz_lab.cli task new --direction verification --title "Inspect stopping-time records" --kind experiment --description "Baseline sweep on a modest interval"
```

Queue a run:

```powershell
python -m collatz_lab.cli run start --direction verification --name "cpu-queued" --start 1 --end 5000 --kernel cpu-parallel --hardware cpu --enqueue-only
```

Validate a run (use the run id from the lab, not a claim id):

```powershell
python -m collatz_lab.cli validate <run_id>
```

Create and link a claim:

```powershell
python -m collatz_lab.cli claim new --direction lemma-workspace --title "Parity clustering candidate" --statement "Odd acceleration may concentrate record-breakers in sparse residue classes."
python -m collatz_lab.cli claim link-run COL-0003 COL-0002 --relation supports
```

Start a worker manually:

```powershell
python -m collatz_lab.cli worker capabilities
python -m collatz_lab.cli worker start --name "local-worker" --hardware auto
```

## Continuous integration

On push and pull requests to `main`, GitHub Actions runs:

- backend: `pip install -e ".[dev]"` and `pytest`
- dashboard: `npm ci` and `npm run build`

See [`.github/workflows/ci.yml`](./.github/workflows/ci.yml).

## Open source notes

- `.env` is ignored and meant for local secrets only
- generated runtime data in `.runtime/`, `data/`, `artifacts/`, and `reports/` may be local-only depending on what you want to publish
- if you open source the repo, review those folders before pushing

## Roadmap

The higher-level plan lives in:

- [`research/ROADMAP.md`](./research/ROADMAP.md)
- [`research/DEVELOPMENT_BACKLOG.md`](./research/DEVELOPMENT_BACKLOG.md)
- [`research/HARDWARE_AND_KERNELS.md`](./research/HARDWARE_AND_KERNELS.md) — target CPUs/GPUs (x86_64, ARM64, Apple Silicon, WoA, CUDA/Metal/ROCm, etc.) and phased kernel portability
- [`research/WORKER_QUEUE.md`](./research/WORKER_QUEUE.md)
- [`research/FEDERATED_LAB.md`](./research/FEDERATED_LAB.md)

## Contributing

Suggestions are welcome, especially on:

- better validation strategies
- lemma falsification workflows
- source review tooling
- GPU kernel design
- UI clarity for evidence vs theory
- ways to make the project more useful for real experimental math

If you open an issue or PR, it helps if you are explicit about which of these you are touching:

- compute / kernels
- validation
- claims / research workflow
- source review
- dashboard UX

## License

Released under the [Apache License 2.0](./LICENSE).
