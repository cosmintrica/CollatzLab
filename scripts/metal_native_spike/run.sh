#!/usr/bin/env bash
# Build CollatzSpike.metallib and the Swift driver, then run the benchmark.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if ! command -v xcrun >/dev/null 2>&1; then
  echo "error: xcrun not found — install Xcode Command Line Tools." >&2
  exit 1
fi

echo "== Compiling Metal ==" >&2
xcrun -sdk macosx metal -c CollatzSpike.metal -o CollatzSpike.air
xcrun -sdk macosx metallib CollatzSpike.air -o CollatzSpike.metallib

echo "== Compiling Swift driver ==" >&2
swiftc -O -framework Metal -framework Foundation main.swift -o metal_native_spike

echo "== Running spike ==" >&2
exec ./metal_native_spike "$@"
