# Dev stack safety: macOS vs Windows

Collatz Lab separates **(A) process orchestration** (how API / Vite / workers are started) from **(B) run safety** (checkpoints, resume, failures). **(B) is cross-platform** and identical wherever the same Python code runs.

## SQLite and multiple workers / API

| Topic | Behavior |
|-------|----------|
| **WAL** | `journal_mode=WAL` — readers (API) and writers (workers, checkpoint thread) overlap more safely than rollback journals alone. |
| **busy_timeout** | Writers wait up to **30s** on lock contention instead of failing immediately with `database is locked`. |
| **Claiming runs** | `claim_next_run` uses **`BEGIN IMMEDIATE`** so only one connection at a time can pass the “pick a QUEUED run + mark RUNNING” sequence — avoids two workers claiming the same row. |
| **Enqueue fill** | When the queue is empty, workers call `queue_continuous_verification_runs` / snacks / random probes. That path is wrapped in **`_serialized_maintenance_enqueue`**: a **file lock** next to the DB (`.collatz_maintenance_enqueue.lock`, `fcntl` on Unix, `filelock` on Windows if installed) so **two processes** cannot both decide “no active GPU” and insert **two** autopilot GPU runs. |
| **Checkpoints** | `execute_run` uses a **single-thread** async writer (`_CheckpointWriter`) so at most one `update_run` is in flight per process; SQLite still serializes commits across connections. |

This keeps the **local** multi-worker + API setup correct; it is not a substitute for a server DB if you shard many hosts on one file.

## Cross-platform (Python — Windows, macOS, Linux)

| Mechanism | What it does |
|-----------|----------------|
| **Checkpoints** | `execute_run` writes `checkpoint` + `metrics` to SQLite on a cadence; async writer drains before the next batch. |
| **Resume** | Next worker claim continues from `checkpoint.next_value` / `last_processed`. |
| **Orphan requeue** | `LabRepository.init()` calls `requeue_orphaned_runs()`: runs left `running` after a crash/restart go back to `queued` with a recovery note (same DB on all OSes). |
| **Worker backoff** | `worker.py`: consecutive failures → exponential backoff (cap 300s); stops after `MAX_CONSECUTIVE_FAILURES` (5) until manual fix. |
| **Overflow guard** | Failed overflow-guard runs trigger `_ensure_overflow_recovery_runs` (prefix/tail recovery queues). |
| **Validation** | Failed validation paths and legacy annotation are OS-agnostic. |

None of the above depends on PowerShell vs bash.

## Orchestration only (differs by script)

| Topic | Windows (`scripts/dev-stack.ps1`) | macOS (`scripts/mac-dev-stack.sh`) |
|-------|-------------------------------------|-------------------------------------|
| Start API + Vite | Yes | Yes |
| Workers | Optional **two** workers (`managed-worker-cpu` + `managed-worker-gpu`) when `-WithWorker` | **Two** managed workers by default: `mac-managed-worker-cpu` + `mac-managed-worker-gpu` (same idea as Windows). Set `COLLATZ_MAC_STACK_SINGLE_WORKER=1` for one `auto` worker (legacy). |
| Health | TCP wait on ports 8000 / 5173 | HTTP `/health` + **PID alive** (avoids false “healthy” on wrong process) |
| Busy API | N/A (implicit) | If Collatz already on :8000, **reuse** API; `stop` may leave external API (`backend_pid=0`) |
| Stop / orphans | Stops tracked processes | `bash scripts/mac-dev-stack.sh stop` uses **`.runtime/mac-dev-stack.json`**. If that file is missing (crash, manual kill, deleted state), plain `stop` has **no PIDs to kill** — the script then prints **who still listens on :8000 / :5173** and suggests `stop force`. **`stop force`** kills **every** listener on those two ports (only if nothing else needs them). |
| Logs | `.runtime/*.log` pattern | Same idea under `.runtime/` |

### Dashboard “Stop compute” vs stopping processes

- **Stop compute** (Compute paused / `continuous_enabled` off) only stops **autopilot from enqueueing new continuous runs**. The **API and worker keep running**; queued work can still be processed. This is intentional.
- To **shut down** API, Vite, and the managed worker on macOS, use **`bash scripts/mac-dev-stack.sh stop`** (or `bootstrap-macos.sh` if it wraps the same), and **`stop force`** if ports stay busy.

### macOS: GPU usage in the dashboard vs Activity Monitor

- The API can report **NVIDIA** utilization via `nvidia-smi`. **Apple Silicon (MPS)** has **no equivalent** in user space, so the dashboard may show **no GPU %** even while **Metal/MPS is busy**.
- **Trust Activity Monitor → CPU tab → View → Columns → “GPU %”** on the **Python** process that runs `collatz_lab.cli worker` (often one process is high CPU+GPU, the API process is lighter).
- The log line **“Recovered after worker restart…”** is added when a **run was left `running` without an active worker claim** (e.g. after you restart the worker). It should **not** repeat the same sentence many times after a code fix; multiple **distinct** restarts can still enqueue recovery once each.

## What is *not* automatic (any OS)

- **No infinite supervisor loop** in-repo: if uvicorn or Vite **exit**, they stay dead until you run `stack:start` / `bootstrap-macos.sh` again (or an external process manager).
- **GPU driver / PyTorch / CUDA** environment issues are not auto-healed by scripts.

For “always on” production, use **systemd**, **launchd**, **pm2**, or Windows **Task Scheduler** / a service wrapper — outside this repository’s scope.

## Compute profile throttling (all OSes — easy to mistake for “slow GPU”)

The dashboard **compute budget** sliders persist to SQLite (`compute_profile`). The worker calls `_compute_budget_throttle_seconds` after each batch: if **effective** system × lane % is below 100%, it **`sleep()`s** so average duty cycle matches the slider. That can reduce throughput to a few M/s while the GPU is idle most of the time.

- **Bypass (any worker):** set environment variable `COLLATZ_SKIP_COMPUTE_THROTTLE=1` (`true` / `yes` also accepted).
- **macOS dev stack:** `mac-dev-stack.sh` exports `COLLATZ_SKIP_COMPUTE_THROTTLE=1` by default for the managed worker so local runs are not accidentally throttled by old slider values. Use `COLLATZ_SKIP_COMPUTE_THROTTLE=0` to honor sliders again.
