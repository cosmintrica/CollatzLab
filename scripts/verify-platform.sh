#!/usr/bin/env bash
# Verify the environment before starting the stack (API + dashboard + worker).
# Run from repo root or anywhere (script resolves ROOT).
#
#   bash scripts/verify-platform.sh              # quick: venv, import, init, UI build, FastAPI factory smoke
#   bash scripts/verify-platform.sh --full       # + pytest on all backend/tests
#   bash scripts/verify-platform.sh --no-dashboard-build
#   bash scripts/verify-platform.sh --check-api  # if API already up: curl /health + /api/directions
#
# Linux / macOS. On Windows see scripts/verify-platform.ps1 or Git Bash.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

FULL=0
SKIP_BUILD=0
CHECK_API=0
for arg in "$@"; do
  case "$arg" in
    --full) FULL=1 ;;
    --no-dashboard-build) SKIP_BUILD=1 ;;
    --check-api) CHECK_API=1 ;;
    -h|--help)
      head -n 20 "$0" | tail -n +2
      exit 0
      ;;
  esac
done

export COLLATZ_LAB_ROOT="$ROOT"
export PYTHONPATH="${ROOT}/backend/src${PYTHONPATH:+:$PYTHONPATH}"

echo "== Collatz Lab — platform check (ROOT=$ROOT) =="

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "ERROR: missing .venv — on macOS: bash bootstrap-macos.sh (or bash scripts/setup-venv.sh + pip install -e \"backend[dev,...]\")." >&2
  exit 1
fi
PY="$ROOT/.venv/bin/python"

echo "[1/5] Import collatz_lab ..."
"$PY" -c "import collatz_lab; print('  OK')"

echo "[2/5] collatz_lab.cli init (idempotent) ..."
"$PY" -m collatz_lab.cli init

echo "[3/5] FastAPI app build smoke (no server) ..."
"$PY" -c "
from collatz_lab.config import Settings
from collatz_lab.api import create_app
app = create_app(Settings.from_env())
print('  OK routes:', len(app.routes))
"

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  echo "[4/5] Dashboard: npm run dashboard:build ..."
  if ! command -v npm >/dev/null 2>&1; then
    echo "ERROR: npm not on PATH (install Node.js)." >&2
    exit 1
  fi
  npm run dashboard:build
else
  echo "[4/5] Dashboard build skipped (--no-dashboard-build)."
fi

if [[ "$FULL" -eq 1 ]]; then
  echo "[5/5] pytest backend/tests ..."
  "$PY" -m pytest backend/tests -q --tb=short
else
  echo "[5/5] pytest skipped (use --full for all tests)."
fi

if [[ "$CHECK_API" -eq 1 ]]; then
  HEALTH_URL="${COLLATZ_LAB_HEALTH_URL:-http://127.0.0.1:8000/health}"
  BASE_URL="${HEALTH_URL%/health}"
  if command -v curl >/dev/null 2>&1; then
    if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
      echo "API: /health OK"
      if curl -sf "${BASE_URL}/api/directions" >/dev/null 2>&1; then
        echo "API: /api/directions OK (looks like Collatz Lab)"
      else
        echo "WARNING: /health OK but /api/directions failed — another service on the same port?" >&2
      fi
    else
      echo "API: no response at $HEALTH_URL (expected if the stack is not running)." >&2
    fi
  else
    echo "WARNING: curl missing — cannot check API." >&2
  fi
fi

echo ""
echo "Check succeeded. Start the stack:"
echo "  macOS:    bash bootstrap-macos.sh"
echo "  npm:      npm run stack:mac:start"
echo "  manual:   bash scripts/run-backend.sh  |  bash scripts/run-dashboard.sh vite  |  bash scripts/run-worker.sh"
