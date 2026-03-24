"""Run validation logic — independent replay and selective cross-checking.

This module contains everything related to verifying that a completed run's
aggregate metrics are correct.  It is intentionally separated from the compute
dispatch (``services.py``) so that validation logic can be read, tested, and
changed without touching the JIT kernels or the execution orchestration.

Validation levels
-----------------
1. **Full replay** (small runs, ≤ ``SELECTIVE_VALIDATION_THRESHOLD`` seeds):
   re-computes the full range using an independent reference and compares.

2. **Selective validation** (large runs):
   - Verifies every record-breaking seed against Python-bigint SoT.
   - Samples ``VALIDATION_WINDOW_COUNT`` random windows and cross-checks.

3. **Coverage gap check** (all runs, verification direction only):
   detects holes in the contiguous verified prefix.

Validation asymmetry note
-------------------------
``_compute_descent_odd_only_reference`` is a *Numba-vs-Numba* oracle (see its
docstring for details).  The true cross-backend SoT net is
``tests/test_differential_cross_backend.py``.
"""

from __future__ import annotations

import logging
import random as _random
from dataclasses import asdict

from .hardware import (
    CPU_ACCELERATED_KERNEL,
    CPU_BARINA_KERNEL,
    CPU_DIRECT_KERNEL,
    CPU_PARALLEL_KERNEL,
    CPU_PARALLEL_ODD_KERNEL,
    CPU_SIEVE_KERNEL,
    GPU_KERNEL,
    GPU_SIEVE_KERNEL,
)
from .repository import LabRepository, utc_now
from .schemas import ArtifactKind, Run, RunStatus
from .metrics_sot import (
    AggregateMetrics,
    INT64_MAX,
    NumberMetrics,
    metrics_descent_direct,
    metrics_direct,
)

logger = logging.getLogger("collatz_lab.validation")

SELECTIVE_VALIDATION_THRESHOLD = 10_000_000
VALIDATION_WINDOW_COUNT = 10
VALIDATION_WINDOW_SIZE = 10_000


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _aggregate_validation_payload(metrics: AggregateMetrics) -> dict:
    return {
        "processed": metrics.processed,
        "last_processed": metrics.last_processed,
        "max_total_stopping_time": metrics.max_total_stopping_time,
        "max_stopping_time": metrics.max_stopping_time,
        "max_excursion": metrics.max_excursion,
    }


def _validation_mode_label(run: Run, *, range_size: int) -> str:
    base = "full replay" if range_size <= SELECTIVE_VALIDATION_THRESHOLD else "selective"
    if run.kernel == CPU_SIEVE_KERNEL:
        return f"{base}, odd-seed descent reference"
    if run.kernel == GPU_SIEVE_KERNEL:
        return f"{base}, descent reference"
    if run.kernel == CPU_BARINA_KERNEL:
        return f"{base}, experimental Barina audit"
    return base


# ---------------------------------------------------------------------------
# Reference aggregates (what each kernel's window should match)
# ---------------------------------------------------------------------------


def _compute_odd_only_reference(
    start: int, end: int, *, sample_limit: int = 12,
) -> AggregateMetrics:
    """Compute reference aggregate for odd seeds only, using metrics_direct."""
    first_odd = start if start & 1 else start + 1
    max_total: dict = {"n": first_odd, "value": -1}
    max_stopping: dict = {"n": first_odd, "value": -1}
    max_exc: dict = {"n": first_odd, "value": -1}
    sample_records: list[dict] = []
    processed = 0
    for n in range(first_odd, end + 1, 2):
        m = metrics_direct(n)
        processed += 1
        if m.total_stopping_time > max_total["value"]:
            max_total = {"n": n, "value": m.total_stopping_time}
            if len(sample_records) < sample_limit:
                sample_records.append({"metric": "max_total_stopping_time", **max_total})
        if m.stopping_time > max_stopping["value"]:
            max_stopping = {"n": n, "value": m.stopping_time}
            if len(sample_records) < sample_limit:
                sample_records.append({"metric": "max_stopping_time", **max_stopping})
        if m.max_excursion > max_exc["value"]:
            max_exc = {"n": n, "value": m.max_excursion}
            if len(sample_records) < sample_limit:
                sample_records.append({"metric": "max_excursion", **max_exc})
    return AggregateMetrics(
        processed=processed,
        last_processed=end,
        max_total_stopping_time=max_total,
        max_stopping_time=max_stopping,
        max_excursion=max_exc,
        sample_records=sample_records,
    )


def _compute_descent_odd_only_reference(
    start: int, end: int, *, sample_limit: int = 12,
) -> AggregateMetrics:
    """Compute a descent-semantics aggregate for odd seeds only (window validation reference).

    **Validation asymmetry — read before changing this function:**

    This is a *Numba-vs-Numba* consistency oracle, not a Python bigint SoT cross-check.
    It delegates to ``compute_range_metrics_parallel_descent_odd`` (another Numba JIT
    kernel), so it catches implementation drift between the two Numba paths but *cannot*
    catch a systematic bug shared by both.

    Cross-backend SoT validation for cpu-sieve / gpu-sieve is covered by:
      - ``_validate_record_seeds``:  per record-breaking seed via ``metrics_descent_direct``
        (Python bigint, arbitrary precision) — true SoT cross-check.
      - ``tests/test_differential_cross_backend.py``:  Python ``sieve_reference`` vs all
        fast backends on the same small interval — CI safety net for the systematic-bug gap.

    If you need this function to become a true bigint SoT reference (slower, but exact for
    all seeds), replace the body with a Python loop over ``metrics_descent_direct``.
    """
    # Lazy import — services.py imports from metrics_sot (which we import from).
    # Keeping this lazy avoids a circular import at module load time.
    from .services import compute_range_metrics_parallel_descent_odd
    return compute_range_metrics_parallel_descent_odd(
        start,
        end,
        sample_limit=sample_limit,
    )


def _kernel_reference_metrics(run: Run, seed: int) -> NumberMetrics | None:
    if run.kernel in {CPU_DIRECT_KERNEL, CPU_ACCELERATED_KERNEL, CPU_PARALLEL_KERNEL, GPU_KERNEL, CPU_PARALLEL_ODD_KERNEL}:
        return metrics_direct(seed)
    if run.kernel in {CPU_SIEVE_KERNEL, GPU_SIEVE_KERNEL}:
        return metrics_descent_direct(seed)
    if run.kernel == CPU_BARINA_KERNEL:
        return None
    return metrics_direct(seed)


def _reference_aggregate_for_kernel(
    run: Run,
    start: int,
    end: int,
) -> AggregateMetrics | None:
    if run.kernel == CPU_PARALLEL_ODD_KERNEL:
        return _compute_odd_only_reference(start, end)
    if run.kernel == CPU_SIEVE_KERNEL:
        return _compute_descent_odd_only_reference(start, end)
    if run.kernel == GPU_SIEVE_KERNEL:
        # gpu-sieve computes odd seeds only (same contract as cpu-sieve); do not use
        # compute_range_metrics_parallel_descent (full contiguous range) or validation
        # will false-fail on processed / max records when the window contains evens.
        return _compute_descent_odd_only_reference(start, end)
    if run.kernel in {CPU_PARALLEL_KERNEL, GPU_KERNEL}:
        from .services import compute_range_metrics  # lazy — see module note
        return compute_range_metrics(start, end, kernel=CPU_DIRECT_KERNEL)
    if run.kernel in {CPU_DIRECT_KERNEL, CPU_ACCELERATED_KERNEL}:
        return None
    if run.kernel == CPU_BARINA_KERNEL:
        return None
    from .services import compute_range_metrics  # lazy
    return compute_range_metrics(start, end, kernel=CPU_DIRECT_KERNEL)


# ---------------------------------------------------------------------------
# Validation strategies
# ---------------------------------------------------------------------------


def _validate_full_replay(run: Run) -> tuple[list[str], list[str]]:
    """Full independent replay used for small runs."""
    mismatches: list[str] = []
    details: list[str] = [
        f"- Mode: full replay ({run.range_end - run.range_start + 1:,} seeds)",
    ]

    if run.kernel in {CPU_DIRECT_KERNEL, CPU_ACCELERATED_KERNEL}:
        from .services import metrics_accelerated  # lazy
        for value in range(run.range_start, run.range_end + 1):
            direct = metrics_direct(value)
            accelerated = metrics_accelerated(value)
            if direct != accelerated:
                mismatches.append(
                    f"{value}: direct={asdict(direct)} accelerated={asdict(accelerated)}"
                )
                break
        details.append("- Compared metrics_direct vs metrics_accelerated per-seed")
        return mismatches, details

    if run.kernel == CPU_BARINA_KERNEL:
        mismatches.append(
            "cpu-barina uses compressed domain metrics and is not eligible for "
            "standard value-for-value validation yet."
        )
        details.append(
            "- Validation skipped: cpu-barina is experimental and its step/excursion metrics "
            "are not semantically identical to standard Collatz."
        )
        return mismatches, details

    from .services import compute_range_metrics  # lazy
    reference = _reference_aggregate_for_kernel(run, run.range_start, run.range_end)
    candidate = compute_range_metrics(run.range_start, run.range_end, kernel=run.kernel)
    if reference is None:
        mismatches.append(f"No reference aggregate is defined for kernel {run.kernel}.")
    elif _aggregate_validation_payload(reference) != _aggregate_validation_payload(candidate):
        mismatches.append(
            f"aggregate mismatch: reference={_aggregate_validation_payload(reference)} "
            f"candidate={_aggregate_validation_payload(candidate)}"
        )

    if run.kernel == CPU_PARALLEL_ODD_KERNEL:
        details.append(f"- Compared {run.kernel} vs per-seed metrics_direct (odd seeds only)")
    elif run.kernel == CPU_SIEVE_KERNEL:
        details.append(f"- Compared {run.kernel} vs standard-Collatz descent reference (odd seeds only)")
    elif run.kernel == GPU_SIEVE_KERNEL:
        details.append(
            f"- Compared {run.kernel} vs standard-Collatz descent reference (odd seeds only, same as cpu-sieve)"
        )
    else:
        details.append(f"- Compared {run.kernel} vs cpu-direct on full range")
    return mismatches, details


def _validate_record_seeds(run: Run) -> tuple[list[str], list[str]]:
    """Verify every record-breaking seed stored in the run's metrics."""
    mismatches: list[str] = []
    details: list[str] = []
    metrics = run.metrics or {}

    if run.kernel == CPU_BARINA_KERNEL:
        details.append("- Record seeds verified: skipped for experimental cpu-barina semantics")
        return mismatches, details

    # Verify the three top-level record holders
    record_keys = {
        "max_total_stopping_time": "total_stopping_time",
        "max_stopping_time": "stopping_time",
        "max_excursion": "max_excursion",
    }
    verified_seeds: set[int] = set()

    def _matches_expected(attr: str, expected_value: int, actual: int) -> bool:
        # Some kernels temporarily cap stored excursion sample records at int64
        # max while retaining the exact top-level record via overflow patches.
        # Validation should treat those capped sample entries as "actual >= cap",
        # not as a hard mismatch.
        if attr == "max_excursion" and expected_value == INT64_MAX:
            return actual >= INT64_MAX
        return actual == expected_value

    for key, attr in record_keys.items():
        record = metrics.get(key)
        if not record or not isinstance(record, dict):
            continue
        n = record.get("n")
        expected_value = record.get("value")
        if n is None or expected_value is None:
            continue
        ref = _kernel_reference_metrics(run, n)
        if ref is None:
            continue
        actual = getattr(ref, attr)
        verified_seeds.add(n)
        if not _matches_expected(attr, expected_value, actual):
            mismatches.append(
                f"Record {key}: seed {n} expected={expected_value} got={actual}"
            )

    # Verify seeds from sample_records
    for record in metrics.get("sample_records", []):
        n = record.get("n")
        metric_name = record.get("metric")
        expected_value = record.get("value")
        if n is None or expected_value is None or n in verified_seeds:
            continue
        ref = _kernel_reference_metrics(run, n)
        if ref is None:
            continue
        verified_seeds.add(n)
        attr = record_keys.get(metric_name)
        if attr is None:
            continue
        actual = getattr(ref, attr)
        if not _matches_expected(attr, expected_value, actual):
            mismatches.append(
                f"Sample record {metric_name}: seed {n} expected={expected_value} got={actual}"
            )

    details.append(f"- Record seeds verified: {len(verified_seeds)}")
    return mismatches, details


def _validate_random_windows(
    run: Run,
    *,
    window_count: int = VALIDATION_WINDOW_COUNT,
    window_size: int = VALIDATION_WINDOW_SIZE,
    rng_seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Sample random windows and compare kernel output vs cpu-direct."""
    mismatches: list[str] = []
    details: list[str] = []
    range_size = run.range_end - run.range_start + 1

    if run.kernel == CPU_BARINA_KERNEL:
        details.append("- Window validation: skipped for experimental cpu-barina semantics")
        mismatches.append(
            "cpu-barina uses compressed domain metrics and is not eligible for "
            "standard window-by-window validation yet."
        )
        return mismatches, details

    def _window_reference(ws: int, we: int) -> AggregateMetrics:
        reference = _reference_aggregate_for_kernel(run, ws, we)
        if reference is None:
            raise ValueError(f"No window reference is defined for kernel {run.kernel}.")
        return reference

    from .services import compute_range_metrics  # lazy
    if range_size <= window_size:
        ref = _window_reference(run.range_start, run.range_end)
        cand = compute_range_metrics(run.range_start, run.range_end, kernel=run.kernel)
        if _aggregate_validation_payload(ref) != _aggregate_validation_payload(cand):
            mismatches.append(
                f"Full-range window mismatch: ref={_aggregate_validation_payload(ref)} "
                f"candidate={_aggregate_validation_payload(cand)}"
            )
        details.append(f"- Window validation: 1 window covering full range")
        return mismatches, details

    rng = _random.Random(rng_seed)
    max_window_start = run.range_end - window_size + 1
    windows_checked = 0

    for _ in range(window_count):
        win_start = rng.randint(run.range_start, max(run.range_start, max_window_start))
        win_end = min(win_start + window_size - 1, run.range_end)
        ref = _window_reference(win_start, win_end)
        cand = compute_range_metrics(win_start, win_end, kernel=run.kernel)
        windows_checked += 1
        if _aggregate_validation_payload(ref) != _aggregate_validation_payload(cand):
            mismatches.append(
                f"Window [{win_start:,}, {win_end:,}] mismatch: "
                f"ref={_aggregate_validation_payload(ref)} "
                f"candidate={_aggregate_validation_payload(cand)}"
            )

    details.append(
        f"- Window validation: {windows_checked} random windows of {window_size:,} seeds each"
    )
    return mismatches, details


def _check_verification_coverage_gaps(
    repository: LabRepository,
    run: Run,
) -> tuple[list[str], list[str]]:
    """Check for gaps in the verification coverage up to and including this run."""
    details: list[str] = []
    warnings: list[str] = []

    if run.direction_slug != "verification":
        details.append("- Coverage gap check: skipped (not a verification run)")
        return warnings, details

    all_runs = repository.list_runs()
    covered = [
        r for r in all_runs
        if r.direction_slug == "verification"
        and r.status in {RunStatus.COMPLETED, RunStatus.VALIDATED}
        and r.id != run.id
    ]
    # Include the run being validated
    covered.append(run)
    covered.sort(key=lambda r: (r.range_start, r.range_end))

    # Merge intervals
    merged: list[tuple[int, int]] = []
    for r in covered:
        if merged and r.range_start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], r.range_end))
        else:
            merged.append((r.range_start, r.range_end))

    # Check if coverage starts from 1
    gaps: list[tuple[int, int]] = []
    if merged and merged[0][0] > 1:
        gaps.append((1, merged[0][0] - 1))

    # Check inter-interval gaps
    for i in range(1, len(merged)):
        if merged[i][0] > merged[i - 1][1] + 1:
            gaps.append((merged[i - 1][1] + 1, merged[i][0] - 1))

    if gaps:
        gap_strs = [f"[{g[0]:,}, {g[1]:,}]" for g in gaps[:5]]
        warnings.append(
            f"Coverage gaps detected: {', '.join(gap_strs)}"
            + (f" (+{len(gaps) - 5} more)" if len(gaps) > 5 else "")
        )

    highest = merged[-1][1] if merged else 0
    details.append(
        f"- Coverage: verified up to {highest:,}, "
        f"{len(gaps)} gap(s) found, "
        f"{len(covered)} completed/validated runs"
    )
    return warnings, details


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def validate_run(
    repository: LabRepository,
    run_id: str,
    *,
    window_count: int = VALIDATION_WINDOW_COUNT,
    window_size: int = VALIDATION_WINDOW_SIZE,
) -> Run:
    run = repository.get_run(run_id)
    range_size = run.range_end - run.range_start + 1
    mismatches: list[str] = []
    report_details: list[str] = [
        f"- Run: {run.id}",
        f"- Range: [{run.range_start:,}, {run.range_end:,}] ({range_size:,} seeds)",
        f"- Kernel: {run.kernel}",
        f"- Hardware: {run.hardware}",
    ]

    if range_size <= SELECTIVE_VALIDATION_THRESHOLD:
        # ── Full replay for small/bounded runs ──
        m, d = _validate_full_replay(run)
        mismatches.extend(m)
        report_details.extend(d)
    else:
        # ── Selective validation for large runs ──
        report_details.append(
            f"- Mode: selective ({window_count} windows of {window_size:,} + record seeds)"
        )

        # 1. Verify every record-breaking seed independently
        m, d = _validate_record_seeds(run)
        mismatches.extend(m)
        report_details.extend(d)

        # 2. Sample random windows and cross-check kernel vs cpu-direct
        m, d = _validate_random_windows(
            run, window_count=window_count, window_size=window_size,
        )
        mismatches.extend(m)
        report_details.extend(d)

    # 3. Check verification coverage gaps (both small and large runs)
    gap_warnings, gap_details = _check_verification_coverage_gaps(repository, run)
    report_details.extend(gap_details)
    if gap_warnings:
        report_details.append(f"- WARNING: {'; '.join(gap_warnings)}")

    # ── Write validation artifact ──
    validation_path = (
        repository.settings.artifacts_dir / "validations" / f"{run.id}-validation.md"
    )
    if mismatches:
        body = (
            f"# Validation for {run.id}\n\n"
            f"Status: **FAILED**\n\n"
            + "\n".join(report_details) + "\n\n"
            f"## Mismatches\n\n"
            + "\n".join(f"- {m}" for m in mismatches) + "\n"
        )
        validation_path.write_text(body, encoding="utf-8")
        repository.create_artifact(
            kind=ArtifactKind.REPORT,
            path=validation_path,
            run_id=run.id,
            metadata={"status": "failed", "mismatches": len(mismatches)},
        )
        return repository.update_run(
            run.id,
            status=RunStatus.FAILED,
            summary=f"Validation failed: {len(mismatches)} mismatch(es) detected.",
            finished_at=utc_now(),
        )

    mode = _validation_mode_label(run, range_size=range_size)
    body = (
        f"# Validation for {run.id}\n\n"
        f"Status: **PASSED** ({mode})\n\n"
        + "\n".join(report_details) + "\n"
    )
    validation_path.write_text(body, encoding="utf-8")
    repository.create_artifact(
        kind=ArtifactKind.REPORT,
        path=validation_path,
        run_id=run.id,
        metadata={"status": "passed", "mode": mode},
    )
    return repository.update_run(
        run.id,
        status=RunStatus.VALIDATED,
        summary=f"Validation passed ({mode}): independent verification confirmed results.",
        finished_at=utc_now(),
    )
