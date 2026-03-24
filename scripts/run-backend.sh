#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

export COLLATZ_LAB_ROOT="$root"
# macOS API process may load Torch + native helpers in one interpreter.
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
backend_src="$root/backend/src"
export PYTHONPATH="${backend_src}${PYTHONPATH:+:$PYTHONPATH}"

# Prefer repo .venv (required for Homebrew Python PEP 668; see scripts/setup-venv.sh).
if [ -x "$root/.venv/bin/python" ]; then
  PY="$root/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python3.13 >/dev/null 2>&1; then
  PY=python3.13
elif command -v python3.12 >/dev/null 2>&1; then
  PY=python3.12
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "error: create .venv first: bash scripts/setup-venv.sh  (or install python3)" >&2
  exit 1
fi

"$PY" -m collatz_lab.main
