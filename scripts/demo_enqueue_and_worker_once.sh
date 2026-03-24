#!/usr/bin/env bash
# Enqueue one small CPU run and execute it once via the worker (SQLite queue).
# Verifies the full path: init → create_run(queued) → worker claims → execute.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

export COLLATZ_LAB_ROOT="$root"
backend_src="$root/backend/src"
export PYTHONPATH="${backend_src}${PYTHONPATH:+:$PYTHONPATH}"

if [ -x "$root/.venv/bin/python" ]; then
  PY="$root/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "error: need .venv (bash scripts/setup-venv.sh) or python3" >&2
  exit 1
fi

echo "Using: $PY"
"$PY" -m collatz_lab.cli init
"$PY" -m collatz_lab.cli run start \
  --direction verification \
  --name demo-local-smoke \
  --start 1 \
  --end 2000 \
  --kernel cpu-parallel \
  --hardware cpu \
  --enqueue-only
"$PY" -m collatz_lab.cli worker once --name demo-worker --hardware auto
echo "OK: one queued run should be completed. Refresh the dashboard Runs ledger."
