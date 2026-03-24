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

- Python 3.13 package: orchestration, SQLite persistence, validation, reports, Reddit feed helpers, hypothesis utilities — see [`docs/HYPOTHESIS_SANDBOX.md`](docs/HYPOTHESIS_SANDBOX.md) for worker cadence, env vars, and **cpu-barina** vs sandbox probes.
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
- **Run rail**, **Reddit intel rail** (r/Collatz intake, not auto-trusted), **live math** navigator / ticker / orbit UI, **research paper** page (KaTeX; `research/paper.json` is generated locally and gitignored)
- Configurable API base via `VITE_API_BASE_URL` for hosted frontends

**Data & research**

- SQLite under `data/` for runs, workers, claims, sources, compute profile, runtime settings (**never committed**)
- `research/`: shared markdown (roadmap, backlog, etc.); exported per-claim notes live under `research/claims/` (**local only**, gitignored)
- Generated outputs: `artifacts/`, `reports/` (gitignored)

**Git pull and your local DB / history:** see [`docs/LOCAL_WORKSPACE_AND_GIT.md`](docs/LOCAL_WORKSPACE_AND_GIT.md) — ignored paths are not overwritten by `git pull`.

**Tooling**

- GitHub Actions CI on `main`: backend `pytest`, dashboard `npm ci` + `npm run build`
- Root `package.json` scripts for the dev stack (PowerShell on Windows); `scripts/*.sh` for backend, dashboard, and worker on Unix-like systems
- **Safety parity (checkpoints / resume / worker backoff):** see [`docs/DEV_STACK_SAFETY.md`](docs/DEV_STACK_SAFETY.md) — run durability is **shared Python code** on Windows and macOS; shell scripts only start processes.
- **Throughput (cpu-sieve vs gpu-sieve / MPS chunking / two workers):** see [`docs/PERFORMANCE_MACOS.md`](docs/PERFORMANCE_MACOS.md). **Why M/s drops over a long run:** [`docs/GPU_SIEVE_THROUGHPUT_STABILITY.md`](docs/GPU_SIEVE_THROUGHPUT_STABILITY.md). **Benchmarks:** `scripts/benchmark_mac_throughput.py`, **MPS sweep:** `scripts/profile_mps_metal_sieve.py`, **Metal chunk sweep + calibration file:** `scripts/profile_metal_sieve_chunk.py` (or from repo root: `npm run bench:metal-chunk` — writes `data/metal_sieve_chunk_calibration.json`). **Future:** benchmark in dashboard + central server + hall of fame — [`docs/ROADMAP_BENCHMARK_PLATFORM.md`](docs/ROADMAP_BENCHMARK_PLATFORM.md). After tuning, `COLLATZ_MAC_EXTREME_THROUGHPUT=1` with `scripts/mac-dev-stack.sh` (see doc). **`COLLATZ_RANDOM_PROBES=0`** disables auto random probe enqueue.

## Research stance

This repo treats the problem conservatively:

- `validated result` means an experiment replay matched through an independent path
- `validated result` does not mean a proof
- external sources are intake material until reviewed
- **Correctness vs fast backends (platform-wide SoT, overflow, test levels):** [`docs/CORRECTNESS_AND_VALIDATION.md`](docs/CORRECTNESS_AND_VALIDATION.md); API summary: `GET /api/validation/contract`
- brute force is treated as evidence and falsification tooling, not the final strategy
- theory work lives in claims, directions, source reviews, and lemma testing
- Gemini can assist with review and planning, but it never becomes the source of truth

## Repository layout

- `backend/`: Python package, CLI, API, services, worker loop, tests
- `dashboard/`: Vite + React UI (`npm` lifecycle inside this folder)
- `research/`: roadmap and direction notes (tracked); `research/claims/` and `research/paper.json` are local exports (gitignored)
- `artifacts/`: generated JSON, Markdown, and evidence outputs
- `reports/`: generated lab reports
- `data/`: SQLite database and runtime state (local)
- `scripts/`: dev stack (PowerShell on Windows), `mac-dev-stack.sh` + root `bootstrap-macos.sh` for macOS, plus shell helpers for API / Vite / workers

## Quickstart

**Before a new session (after `git pull` or dependency changes):** run environment verification, then start the stack — see [`docs/PLATFORM_VERIFY_AND_START.md`](docs/PLATFORM_VERIFY_AND_START.md).

- macOS/Linux: `bash scripts/verify-platform.sh` (or `bash bootstrap-macos.sh verify`)
- with full tests: `bash scripts/verify-platform.sh --full`
- Windows: `powershell -ExecutionPolicy Bypass -File .\scripts\verify-platform.ps1`

### macOS: automatic stack (clone → run)

On **macOS**, a single script creates the venv, installs Python deps (on **Apple Silicon** it adds **`mps`** so PyTorch/MPS is available), installs dashboard `npm` packages, runs `init`, starts the **API** (or **reuses** Collatz Lab if it already answers on port 8000), starts **Vite** and a **worker**. If the database has no runs yet, it enqueues one tiny CPU demo job so the dashboard ledger is not empty.

**Requirements:** Python 3.11+ on `PATH`, Node.js (`node` + `npm`), `curl`.

```bash
cd CollatzLab
bash bootstrap-macos.sh
```

- Dashboard: `http://127.0.0.1:5173/`
- API health: `http://127.0.0.1:8000/health`
- Stop: `bash bootstrap-macos.sh stop` — or `bash scripts/mac-dev-stack.sh stop`. If nothing seems to stop (missing state file or orphan listeners), run `bash scripts/mac-dev-stack.sh stop force` (frees **:8000** and **:5173**).
- Status: `bash bootstrap-macos.sh status`

Optional environment variables (same shell session as `start`):

- `COLLATZ_MAC_STACK_NO_WORKER=1` — do not start any workers (runs stay queued until you start one manually).
- `COLLATZ_MAC_STACK_SINGLE_WORKER=1` — start one worker with `--hardware auto` instead of the default **CPU + GPU** pair (same queue model as Windows `-WithWorker`, which starts two workers).
- **`COLLATZ_STACK_CPU_WORKERS` / `COLLATZ_STACK_GPU_WORKERS`** — on **macOS** (`mac-dev-stack.sh`) and **Windows** (`dev-stack.ps1 -WithWorker`): how many managed **CPU** / **GPU** worker processes to start (default **1** each). CPU workers get a **split `NUMBA_NUM_THREADS`** to reduce oversubscription. On a **single** Apple GPU, extra GPU workers rarely help.
- `COLLATZ_MAC_STACK_NO_SEED=1` — do not enqueue the welcome demo run on an empty DB.
- `COLLATZ_MAC_STACK_ALLOW_BUSY_API=1` — if `/health` responds but `/api/directions` is **not** recognized as Collatz Lab, try to spawn a second API anyway (usually fails to bind; debugging only). When **Collatz Lab** is already on port 8000, the script **reuses** it and only starts the worker + Vite.
- **`COLLATZ_SKIP_COMPUTE_THROTTLE`** — `bootstrap-macos.sh` / `mac-dev-stack.sh` set this to **`1` by default** for the worker so dashboard **compute sliders** (stored in SQLite) do not add `sleep()` after every batch (which can make GPU runs look “stuck” at a few M/s). Set to `0` before `start` if you want throttling back.
- **`COLLATZ_MAC_EXTREME_THROUGHPUT=1`** — for Apple GPU workers: if unset, sets **2097152** / **128** (larger MPS chunk, same sync as profiling; see `docs/PERFORMANCE_MACOS.md`).

From npm: `npm run stack:mac:start` / `stack:mac:stop` / `stack:mac:status` / `stack:mac:restart` (requires `bash` on PATH).

**Troubleshooting**

- **`externally-managed-environment` (Homebrew Python):** do not `pip install` into Homebrew’s interpreter. This flow always uses **`./.venv`** — if you deleted it, run `bash bootstrap-macos.sh` again.
- **Collatz Lab API already on port 8000:** `bootstrap-macos.sh` **reuses** it (detects `/api/directions`) and only starts **workers** + **Vite**. `mac-dev-stack.sh stop` stops **both** managed workers + dashboard; it **does not kill** the API if this script did not start it (`backend_pid=0` in `status`). **`stop force`** kills listeners on :8000 and :5173 (avoid if another app needs those ports). If another app owns the port, you get an error — use `lsof -nP -iTCP:8000 -sTCP:LISTEN` or change the port in `.env`.
- **Dashboard “Stop compute”:** pauses **continuous autopilot enqueue** only; it does **not** exit API/worker/Vite — use the stack **stop** commands above for that.

---

1. **Python:** 3.11+ is supported; **3.13** is recommended (same as CI).

2. Install backend dependencies:

   **macOS (Homebrew Python — PEP 668):** Homebrew’s `python3.13` refuses `pip install` into the system interpreter. Use a project venv:

   ```bash
   bash scripts/setup-venv.sh
   source .venv/bin/activate
   ```

   Optional GPU extras (NVIDIA/CUDA wheels — same as Windows/Linux):

   ```bash
   .venv/bin/pip install -e "backend[dev,gpu]"
   ```

   Optional **host metrics** (`psutil`) for smoother CPU usage on macOS and consistent sampling across OSes (without it, Linux still uses `/proc/stat`; Windows uses performance counters):

   ```bash
   .venv/bin/pip install -e "backend[dev,system]"
   ```

   **Apple Silicon GPU runs** (`gpu-collatz-accelerated` / `gpu-sieve` in the worker) need **PyTorch with MPS**:

   ```bash
   .venv/bin/pip install -e "backend[dev,mps]"
   ```

   `scripts/run-backend.sh` uses `.venv/bin/python` automatically when `.venv` exists.

   **Apple Silicon / Metal throughput (optional env, worker process):** PyTorch drives the GPU; larger micro-batches mean fewer CPU↔GPU syncs (unified memory helps, but very large values can spike RAM). Defaults are tuned for M-series chips.

   - `COLLATZ_MPS_BATCH_SIZE` — seeds per MPS chunk for **`gpu-collatz-accelerated`** (default `32768`, clamped 1024…2 097 152).
   - `COLLATZ_MPS_SIEVE_BATCH_SIZE` — odd seeds per MPS sub-chunk for **`gpu-sieve`** (default **1048576**, clamped 4096…4194304; tune with `scripts/profile_mps_metal_sieve.py`).
   - **`COLLATZ_METAL_SIEVE_CHUNK_*` + calibration** — native Metal **`gpu-sieve`**: explicit `COLLATZ_METAL_SIEVE_CHUNK_SIZE`, cap, **`COLLATZ_METAL_SIEVE_CHUNK_AUTO`** (default **on**). Auto prefers **`data/metal_sieve_chunk_calibration.json`** from `profile_metal_sieve_chunk.py --write-calibration` (throughput), then RAM ladder; TTL `COLLATZ_METAL_SIEVE_CALIBRATION_MAX_AGE_DAYS` (default 30); `COLLATZ_METAL_SIEVE_USE_CALIBRATION=0` skips the file. **`npm run bench:metal-chunk`** / **`--print-auto`**. **`COLLATZ_METAL_SIEVE_STDIO_PIPELINE`** — default **1**. Stdio overhead: `scripts/benchmark_metal_stdio_overhead.py`.
   - `COLLATZ_MPS_SYNC_EVERY` — how many outer Collatz steps to run on GPU **before** one `any()` check to exit early (default `24`, clamped 1…2048). Lower values mean more CPU↔GPU syncs and often **lower throughput**; try `32`–`64` if you want slightly less host overhead (uses a bit more GPU work after seeds finish).

   CPU parallelism for Numba kernels still follows `NUMBA_NUM_THREADS` and the dashboard **compute profile** (100% system + CPU/GPU lanes = no artificial idle).

   **Native stack (macOS):** `brew install libomp`, then `bash scripts/mac-dev-stack.sh build-natives` (CPU `.dylib` + `metal_sieve_chunk`). Start with optional `COLLATZ_MAC_STACK_BUILD_NATIVES=1` to rebuild on each `start`. Defaults: `COLLATZ_CPU_SIEVE_BACKEND=auto`, `COLLATZ_GPU_SIEVE_BACKEND=auto`. Check `GET /api/workers/native-stack`. Full checklist: [`docs/MACOS_FULL_NATIVE_CPU_AND_METAL_GPU.md`](docs/MACOS_FULL_NATIVE_CPU_AND_METAL_GPU.md) and [`docs/CPU_SIEVE_NATIVE_BACKEND.md`](docs/CPU_SIEVE_NATIVE_BACKEND.md).

   **Activity Monitor:** macOS often does **not** show a clear “GPU %” for Metal/MPS the way Task Manager does for CUDA; low host CPU with a fast seed rate usually still means the GPU is doing the heavy lifting.

   **Windows (PowerShell):**

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

   ```bash
   npm install --prefix dashboard
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

**macOS / Linux (no PowerShell)** — from the repo root, after `bash scripts/setup-venv.sh` (or your own venv) and `npm install --prefix dashboard`:

```bash
export COLLATZ_LAB_ROOT="$PWD"
python3 -m collatz_lab.cli init
bash scripts/run-backend.sh
bash scripts/run-dashboard.sh vite
```

**Worker vs API:** the FastAPI process only exposes data; **Collatz runs stay `queued` until a worker claims them**. `scripts/run-worker.sh` uses `.venv/bin/python` when present (same as the API). Quick smoke test: `bash scripts/demo_enqueue_and_worker_once.sh` then refresh the dashboard run list.

**Local API keys (e.g. Gemini):** the dashboard can write secrets to `.env` in the workspace. That file is listed in `.gitignore`; keep it local, do not commit it, and treat the machine like any other dev environment (malware, shared accounts, and accidental uploads are still risks).

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

Stuck **`running`** GPU run (keeps checkpoint; see [`docs/RUN_RECOVERY.md`](./docs/RUN_RECOVERY.md)):

```powershell
python -m collatz_lab.cli run release COL-0022
python -m collatz_lab.cli run release COL-0022 --migrate-cpu-sieve
```

Or from repo root (bash):

```bash
bash scripts/recover-stuck-run.sh COL-0022 --migrate-cpu-sieve
```

**Append a summary note without changing status** (e.g. validation incident / documentation):

```powershell
python -m collatz_lab.cli run append-summary COL-0254 --text "Your note here."
```

Historical `gpu-sieve` false positives (validator): see [`docs/VALIDATION_INCIDENT_GPU_SIEVE_ODD_REFERENCE.md`](docs/VALIDATION_INCIDENT_GPU_SIEVE_ODD_REFERENCE.md) and `bash scripts/annotate-gpu-sieve-validator-false-positives.sh` (dry-run / `--apply`).

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

**Handoff / current pause:** [docs/STATUS_AND_NEXT_STEPS.md](./docs/STATUS_AND_NEXT_STEPS.md) — Metal spike checklist, Metal Toolchain vs end users, and **GitHub / multi-repo vs monorepo** pointers (links to the federated hosting plan in `research/`).

The higher-level plan lives in:

- [`research/ROADMAP.md`](./research/ROADMAP.md)
- [`research/DEVELOPMENT_BACKLOG.md`](./research/DEVELOPMENT_BACKLOG.md)
- [`research/HARDWARE_AND_KERNELS.md`](./research/HARDWARE_AND_KERNELS.md) — target CPUs/GPUs (x86_64, ARM64, Apple Silicon, WoA, CUDA/Metal/ROCm, etc.) and phased kernel portability
- [`research/SMART_DETECTION.md`](./research/SMART_DETECTION.md) — `metadata.smart_detection` JSON from automatic probes (for UI, tooling, and backend spikes)
- [`research/KERNEL_PORTABILITY_GUARDRAILS.md`](./research/KERNEL_PORTABILITY_GUARDRAILS.md) — non-regression rules for kernel work (Windows x86_64 / AMD CPU / NVIDIA baseline)
- [`research/WORKER_QUEUE.md`](./research/WORKER_QUEUE.md)
- [`research/FEDERATED_LAB.md`](./research/FEDERATED_LAB.md)
- [`research/UI_SURFACE_AND_MONOREPO_PLAN.md`](./research/UI_SURFACE_AND_MONOREPO_PLAN.md) — monorepo OK; **public site vs worker console** (tabs, `VITE_COLLATZ_UI_SHELL`, build phases, API scopes)
- [`research/NATIVE_SIEVE_PORT.md`](./research/NATIVE_SIEVE_PORT.md) — **native Metal + C** parity path for `gpu-sieve` / `cpu-sieve`; kit under [`scripts/native_sieve_kit/`](./scripts/native_sieve_kit/)

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
