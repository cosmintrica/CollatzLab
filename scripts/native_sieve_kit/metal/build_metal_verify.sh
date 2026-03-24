#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if ! command -v xcrun >/dev/null 2>&1; then
  echo "error: xcrun not found" >&2
  exit 1
fi

xcrun -sdk macosx metal -c CollatzLabSieve.metal -o CollatzLabSieve.air
xcrun -sdk macosx metallib CollatzLabSieve.air -o CollatzLabSieve.metallib
swiftc -O -framework Metal -framework Foundation verify_main.swift -o metal_lab_sieve_verify
ln -sf metal_lab_sieve_verify metal_sieve_verify
echo "built metal_lab_sieve_verify (+ symlink metal_sieve_verify) + CollatzLabSieve.metallib"
