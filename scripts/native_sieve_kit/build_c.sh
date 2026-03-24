#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
cc -std=c11 -O3 -Wall -Wextra -o sieve_descent sieve_descent.c
echo "built $ROOT/sieve_descent"
# Optional: same source also builds the lab native cpu-sieve backend (Darwin/Linux only).
if uname -s | grep -qE 'Darwin|Linux'; then
  bash "$ROOT/build_native_cpu_sieve_lib.sh"
fi
