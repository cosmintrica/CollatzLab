#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

export COLLATZ_LAB_ROOT="$root"
backend_src="$root/backend/src"
export PYTHONPATH="${backend_src}${PYTHONPATH:+:$PYTHONPATH}"

python -m collatz_lab.main
