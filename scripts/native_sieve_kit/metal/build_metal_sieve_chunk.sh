#!/usr/bin/env bash
# Builds CollatzLabSieve.metallib + metal_sieve_chunk (production gpu-sieve helper).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if ! command -v xcrun >/dev/null 2>&1; then
  echo "error: xcrun not found" >&2
  exit 1
fi

xcrun -sdk macosx metal -c CollatzLabSieve.metal -o CollatzLabSieve.air
xcrun -sdk macosx metallib CollatzLabSieve.air -o CollatzLabSieve.metallib
swiftc -O -framework Metal -framework Foundation chunk_main.swift -o metal_sieve_chunk
echo "built metal_sieve_chunk + CollatzLabSieve.metallib"
