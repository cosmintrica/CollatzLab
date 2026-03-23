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
- visualizing live checkpoints and derived formulas from real runs.

## Current capabilities

- Python backend for orchestration, persistence, validation, and reports
- FastAPI API for local automation and dashboard access
- React/Vite dashboard with live math, evidence views, queue operations, and source review
- SQLite-backed local state for runs, claims, artifacts, workers, and sources
- CPU worker loop and GPU worker path with resumable checkpoints
- source intake workflow, consensus baseline, and fallacy tagging
- source review history plus Gemini-assisted draft reviews in guarded `review_only` mode
- guarded Gemini autopilot that turns local lab history into bounded task proposals and optional background task creation
- Reddit intake rail for watching new `r/Collatz` threads without treating them as truth

## Research stance

This repo treats the problem conservatively:

- `validated result` means an experiment replay matched through an independent path
- `validated result` does not mean a proof
- external sources are intake material until reviewed
- brute force is treated as evidence and falsification tooling, not the final strategy
- theory work lives in claims, directions, source reviews, and lemma testing
- Gemini can assist with review and planning, but it never becomes the source of truth

## Repository layout

- `backend/`: Python package, CLI, API, services, worker loop, and tests
- `dashboard/`: local web UI built with React and Vite
- `research/`: roadmap, direction notes, and claim documents
- `artifacts/`: generated JSON, Markdown, and evidence outputs
- `reports/`: generated lab reports
- `data/`: SQLite database and runtime state
- `scripts/`: local stack and worker startup scripts

## Quickstart

1. Install backend dependencies:

   ```powershell
   pip install -e .\backend[dev]
   ```

2. Install dashboard dependencies:

   ```powershell
   npm install --prefix .\dashboard
   ```

3. Configure local environment if needed:

   - edit `.env`
   - or copy `.env.example` to `.env`
   - if no Gemini key is detected, the dashboard prompts for it automatically and stores it in the local `.env`

   Example Gemini setup:

   ```text
   COLLATZ_LLM_ENABLED=1
   GEMINI_MODEL=gemini-2.5-flash
   GEMINI_API_KEY=your_key_here
   COLLATZ_LLM_AUTOPILOT_ENABLED=1
   COLLATZ_LLM_AUTOPILOT_INTERVAL_SECONDS=1800
   COLLATZ_LLM_AUTOPILOT_MAX_TASKS=3
   ```

4. Initialize the workspace:

   ```powershell
   python -m collatz_lab.cli init
   ```

5. Start the managed local stack:

   ```powershell
   npm run stack:start:worker
   ```

6. Open:

- dashboard: `http://localhost:5173`
- API health: `http://127.0.0.1:8000/health`

## Vercel frontend hosting

The dashboard is prepared for a frontend-only Vercel deploy.

- `vercel.json` at the repo root points Vercel to `dashboard/`
- the frontend reads `VITE_API_BASE_URL`
- if `VITE_API_BASE_URL` is not set, the site falls back to the local API URL for desktop/local use

For a hosted frontend, set `VITE_API_BASE_URL` in Vercel to the public backend URL you want the dashboard to call.

## Local commands

Managed stack:

```powershell
npm run stack:start
npm run stack:start:worker
npm run stack:start:vite
npm run stack:start:vite:worker
npm run stack:status
npm run stack:stop
npm run stack:restart
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

Validate a run:

```powershell
python -m collatz_lab.cli validate COL-0002
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

## Open source notes

- `.env` is ignored and meant for local secrets only
- generated runtime data in `.runtime/`, `data/`, `artifacts/`, and `reports/` may be local-only depending on what you want to publish
- if you open source the repo, review those folders before pushing

## Roadmap

The higher-level plan lives in:

- [`research/ROADMAP.md`](./research/ROADMAP.md)
- [`research/DEVELOPMENT_BACKLOG.md`](./research/DEVELOPMENT_BACKLOG.md)
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
