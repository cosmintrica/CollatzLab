#!/usr/bin/env bash
# Create a project-local .venv and install the backend (avoids Homebrew PEP 668 errors).
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

if [ -d .venv ]; then
  echo "'.venv' already exists. Activate with:"
  echo "  source .venv/bin/activate"
  echo "To reinstall deps: .venv/bin/pip install -e \"backend[dev]\""
  exit 0
fi

if command -v python3.13 >/dev/null 2>&1; then
  python3.13 -m venv .venv
elif command -v python3 >/dev/null 2>&1; then
  python3 -m venv .venv
else
  echo "error: need python3.13 or python3 on PATH" >&2
  exit 1
fi

.venv/bin/python -m pip install -U pip
.venv/bin/pip install -e "backend[dev]"

echo ""
echo "Virtual environment ready at .venv/"
echo "  source .venv/bin/activate"
echo "  bash scripts/run-backend.sh"
