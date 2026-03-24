# Platform verification and safe startup

Goal: after clone / dependency updates, confirm **backend + UI** are consistent before starting the API and workers.

## Quick checklist

1. **Stop** an old stack if needed: `bash scripts/mac-dev-stack.sh stop` (macOS) or `npm run stack:stop` (Windows).
2. **Verify the environment** (commands below).
3. **Start** the stack: `bash bootstrap-macos.sh` or `npm run stack:start:worker`.

## Verification commands

| Environment | Command |
|-------------|---------|
| **macOS / Linux** | `bash scripts/verify-platform.sh` |
| **+ full pytest** | `bash scripts/verify-platform.sh --full` |
| **API already running** | `bash scripts/verify-platform.sh --check-api` |
| **macOS bootstrap** | `bash bootstrap-macos.sh verify` or `bash bootstrap-macos.sh verify --full` |
| **npm (needs `bash` on PATH)** | `npm run verify:platform` / `npm run verify:platform:full` |
| **Windows PowerShell** | `powershell -ExecutionPolicy Bypass -File .\scripts\verify-platform.ps1` |
| **Windows + pytest** | `...\verify-platform.ps1 -Full` |

What the “quick” check does:

- `.venv` exists with the project Python;
- `import collatz_lab` and `collatz_lab.cli init`;
- `create_app()` instantiates (without binding a port);
- `npm run dashboard:build` (production Vite build).

`--full` adds `pytest backend/tests` (same set as CI).

## After startup

- API: `http://127.0.0.1:8000/health`
- Dashboard dev: `http://127.0.0.1:5173/`
- **Runs stay `queued` without a worker** — the stack with workers starts separate processes; see README § Quickstart.

## Common issues

- **Port 8000 in use:** see `mac-dev-stack.sh` startup message; `lsof -nP -iTCP:8000 -sTCP:LISTEN` or change the port in `.env`.
- **Run stuck in `running`:** [`RUN_RECOVERY.md`](./RUN_RECOVERY.md).
- **Apple GPU:** install extra `backend[dev,mps]` (`bootstrap-macos.sh` does this on Apple Silicon automatically).
