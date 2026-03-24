#!/usr/bin/env bash
# Release a stuck RUNNING run back to queued (keeps checkpoint). Optional: migrate to cpu-sieve.
# Usage (from repo root):
#   bash scripts/recover-stuck-run.sh COL-0022
#   bash scripts/recover-stuck-run.sh COL-0022 --migrate-cpu-sieve --note "GPU hang"
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 1 || "$1" == -* ]]; then
  echo "usage: bash scripts/recover-stuck-run.sh COL-XXXX [args passed to: lab run release ...]" >&2
  echo "  e.g.  bash scripts/recover-stuck-run.sh COL-0022 --migrate-cpu-sieve" >&2
  exit 1
fi

RUN_ID="$1"
shift

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi

export PYTHONPATH="$ROOT/backend/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -m collatz_lab.cli run release "$RUN_ID" "$@"
