#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

mode="${1:-static}"

if [ "$mode" = "vite" ]; then
  npm run dev --prefix ./dashboard -- --host 127.0.0.1
  exit $?
fi

npm run build --prefix ./dashboard
python -m http.server 5173 --bind 127.0.0.1 --directory dashboard/dist
