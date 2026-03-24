# Incident: false-positive validation for `gpu-sieve` (Mar 2026)

## What happened

Some **`gpu-sieve`** runs were marked **FAILED** at validation with messages like:

`Validation failed: 1 mismatch(es) detected.`

The cause was **not** wrong GPU/Metal computation, but a **bug in the validator reference**: for `gpu-sieve` the reference used a **full-interval** aggregate (`compute_range_metrics_parallel_descent`), while the kernel only processes **odd** seeds (same contract as `cpu-sieve`). That made `processed` and sometimes maxima disagree → false positive.

## Code fix

In `services.py`, `_reference_aggregate_for_kernel` for `GPU_SIEVE_KERNEL` now uses **`_compute_descent_odd_only_reference`** (same as `cpu-sieve`).

## Optional: annotate affected runs in the DB

Append an explanatory line to **summary** without changing status:

```bash
export COLLATZ_LAB_ROOT="$PWD"
export PYTHONPATH=backend/src
.venv/bin/python -m collatz_lab.cli run append-summary COL-0254 --text \
  "Superseded by validator fix (odd-only reference for gpu-sieve, 2026-03). GPU compute was not at fault; re-run COL-0259 or re-validate optional."
```

Replace `COL-0254` with each relevant run id (e.g. other failed `random-probe-gpu-sieve` in the same period).

## About COL-0259

If **COL-0259** was an automatic re-validation after that false positive, you can let it finish or cancel per your policy; with the new code, validation for the same interval should not fail for the same reason.
