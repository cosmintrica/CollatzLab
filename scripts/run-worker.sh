#!/usr/bin/env bash
set -euo pipefail

name="${1:-managed-worker}"
hardware="${2:-auto}"
poll_interval="${3:-5}"

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

export COLLATZ_LAB_ROOT="$root"
export COLLATZ_LAB_WORKER_NAME="$name"
export COLLATZ_LAB_WORKER_HARDWARE="$hardware"
export COLLATZ_CPU_PARALLEL_BATCH_SIZE="250000000"
export COLLATZ_CPU_PARALLEL_ODD_BATCH_SIZE="500000000"
export COLLATZ_GPU_BATCH_SIZE="500000000"
export COLLATZ_GPU_THREADS_PER_BLOCK="256"
export NUMBA_NUM_THREADS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

backend_src="$root/backend/src"
export PYTHONPATH="${backend_src}${PYTHONPATH:+:$PYTHONPATH}"

python -m collatz_lab.cli worker start --name "$name" --hardware "$hardware" --poll-interval "$poll_interval"
