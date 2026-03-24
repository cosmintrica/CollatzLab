#!/usr/bin/env bash
# Append a note to FAILED gpu-sieve runs with "Validation failed" (validator fix supersession).
#
#   bash scripts/annotate-gpu-sieve-validator-false-positives.sh           # dry-run
#   bash scripts/annotate-gpu-sieve-validator-false-positives.sh --apply   # apply
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
export COLLATZ_LAB_ROOT="$root"
export PYTHONPATH="${root}/backend/src${PYTHONPATH:+:$PYTHONPATH}"
PY="${root}/.venv/bin/python"
[[ -x "$PY" ]] || PY=python3

NOTE='Superseded by validator fix (gpu-sieve odd-only reference, 2026-03). GPU path not at fault.'

apply=0
[[ "${1:-}" == "--apply" ]] && apply=1

ids="$("$PY" <<'PY'
import os
import sqlite3
from pathlib import Path

db = Path(os.environ["COLLATZ_LAB_ROOT"]) / "data" / "lab.db"
conn = sqlite3.connect(db)
cur = conn.execute(
    """
    SELECT id FROM runs
    WHERE status = 'failed' AND kernel = 'gpu-sieve'
      AND summary LIKE 'Validation failed:%'
      AND summary NOT LIKE '%validator fix (gpu-sieve odd-only%'
    """
)
print(" ".join(r[0] for r in cur.fetchall()))
conn.close()
PY
)"

if [[ -z "$ids" ]]; then
  echo "No matching runs to annotate."
  exit 0
fi

echo "Runs: $ids"
if [[ "$apply" -ne 1 ]]; then
  echo "Dry-run. Re-run with --apply to append summary on each."
  exit 0
fi

for id in $ids; do
  echo "Annotating $id ..."
  "$PY" -m collatz_lab.cli run append-summary "$id" --text "$NOTE"
done
