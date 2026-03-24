#!/usr/bin/env bash
# One entrypoint after cloning on macOS: setup + start API, Vite, and worker.
# Apple Silicon: installs PyTorch (MPS) automatically via scripts/mac-dev-stack.sh.
#
#   bash bootstrap-macos.sh           # default: start
#   bash bootstrap-macos.sh start
#   bash bootstrap-macos.sh stop
#   bash bootstrap-macos.sh stop force   # also free :8000 / :5173 if state file missing
#   bash bootstrap-macos.sh status
#   bash bootstrap-macos.sh restart
#   bash bootstrap-macos.sh verify          # check venv, init, UI build, FastAPI smoke (no full pytest)
#   bash bootstrap-macos.sh verify --full   # + pytest backend/tests
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
ACTION="${1:-start}"
if [[ $# -ge 1 && "$1" =~ ^(start|stop|restart|status|verify)$ ]]; then
  ACTION="$1"
  shift
fi
if [[ "$ACTION" == "verify" ]]; then
  exec bash "$ROOT/scripts/verify-platform.sh" "$@"
fi
exec bash "$ROOT/scripts/mac-dev-stack.sh" "$ACTION" "$@"
