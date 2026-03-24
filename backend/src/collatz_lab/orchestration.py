"""Run execution, checkpoint writing, report generation, and queue dispatch for Collatz Lab.

Extracted from ``services.py``.  Imports compute functions from ``services``
lazily (inside function bodies) to avoid a module-level circular dependency:

    services → orchestration  (re-export at module level, safe)
    orchestration → services  (lazy, inside execute_run body only)
"""
from __future__ import annotations

import json
import logging
import os
import time
import traceback as _traceback_mod
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from ._profile_helpers import _effective_profile_percent, _positive_int_env
from .hardware import (
    CPU_ACCELERATED_KERNEL,
    CPU_BARINA_KERNEL,
    CPU_PARALLEL_KERNEL,
    CPU_PARALLEL_ODD_KERNEL,
    CPU_SIEVE_KERNEL,
    GPU_KERNEL,
    GPU_SIEVE_KERNEL,
)
from .metrics_sot import MAX_KERNEL_STEPS, collatz_step
from .repository import LabRepository, sha256_text, utc_now
from .scheduling import (
    _ensure_overflow_recovery_runs,
    _serialized_maintenance_enqueue,
    annotate_legacy_validation_failures,
    queue_continuous_verification_runs,
    queue_legacy_validation_reruns,
    queue_randomized_compute_runs,
    queue_research_snack_runs,
)
from .schemas import ArtifactKind, ComputeProfile, ModularProbeResult, RunStatus

logger = logging.getLogger("collatz_lab.orchestration")


# ---------------------------------------------------------------------------
# Async checkpoint writer
# ---------------------------------------------------------------------------


class _CheckpointWriter:
    """Async DB checkpoint writer for execute_run.

    Runs ``repository.update_run`` in a single background thread so the
    GPU (or CPU kernel) can start the next batch while SQLite commits
    the previous one.

    At most one write is in-flight at a time.  ``drain()`` blocks until
    the pending write completes.  This guarantees crash-safety: if the
    process dies mid-batch, the *previous* checkpoint is already committed.
    """

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="checkpoint")
        self._pending: Future | None = None

    def drain(self) -> None:
        """Block until the in-flight write finishes.  Re-raises on failure."""
        if self._pending is not None:
            self._pending.result()
            self._pending = None

    def submit(self, repository: LabRepository, run_id: str, **kwargs) -> None:
        """Drain the previous write, then submit a new one."""
        self.drain()
        self._pending = self._executor.submit(repository.update_run, run_id, **kwargs)

    def shutdown(self) -> None:
        """Drain pending work and shut down the thread pool."""
        self.drain()
        self._executor.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Compute-budget throttle
# ---------------------------------------------------------------------------


def _compute_budget_throttle_seconds(
    *,
    hardware: str,
    profile: ComputeProfile | None,
    compute_sec: float,
) -> float:
    """Idle time so average duty cycle matches the compute profile.

    Batch size and CPU thread count both scale with the same effective percent,
    so wall time per batch stays ~flat between 50% and 100% without extra
    idle. GPU batches also keep the device saturated for the whole kernel.
    Sleeping after each batch restores a visible slowdown when the slider is
    below 100%.
    """
    if os.getenv("COLLATZ_SKIP_COMPUTE_THROTTLE", "").strip().lower() in ("1", "true", "yes"):
        return 0.0
    if compute_sec <= 0:
        return 0.0
    lane = "gpu" if hardware == "gpu" else "cpu"
    effective = _effective_profile_percent(profile, lane)
    if effective >= 100:
        return 0.0
    eff = max(1, effective)
    return compute_sec * (100.0 / eff - 1.0)


# ---------------------------------------------------------------------------
# Effective checkpoint interval
# ---------------------------------------------------------------------------


def _effective_checkpoint_interval(
    kernel: str,
    checkpoint_interval: int,
    profile: ComputeProfile | None = None,
) -> int:
    cpu_percent = _effective_profile_percent(profile, "cpu")
    gpu_percent = _effective_profile_percent(profile, "gpu")
    minimums = {
        CPU_ACCELERATED_KERNEL: max(
            50_000,
            round(_positive_int_env("COLLATZ_CPU_ACCELERATED_BATCH_SIZE", 250_000) * max(cpu_percent, 5) / 100),
        ),
        CPU_PARALLEL_KERNEL: max(
            1_000_000,
            round(_positive_int_env("COLLATZ_CPU_PARALLEL_BATCH_SIZE", 50_000_000) * max(cpu_percent, 5) / 100),
        ),
        CPU_PARALLEL_ODD_KERNEL: max(
            2_000_000,
            round(_positive_int_env("COLLATZ_CPU_PARALLEL_ODD_BATCH_SIZE", 100_000_000) * max(cpu_percent, 5) / 100),
        ),
        GPU_KERNEL: max(
            10_000_000,
            round(_positive_int_env("COLLATZ_GPU_BATCH_SIZE", 100_000_000) * max(gpu_percent, 5) / 100),
        ),
        CPU_SIEVE_KERNEL: max(
            5_000_000,
            round(_positive_int_env("COLLATZ_CPU_SIEVE_BATCH_SIZE", 250_000_000) * max(cpu_percent, 5) / 100),
        ),
        CPU_BARINA_KERNEL: max(
            5_000_000,
            round(_positive_int_env("COLLATZ_CPU_BARINA_BATCH_SIZE", 200_000_000) * max(cpu_percent, 5) / 100),
        ),
        GPU_SIEVE_KERNEL: max(
            25_000_000,
            round(_positive_int_env("COLLATZ_GPU_SIEVE_BATCH_SIZE", 500_000_000) * max(gpu_percent, 5) / 100),
        ),
    }
    return max(checkpoint_interval, minimums.get(kernel, checkpoint_interval))


# ---------------------------------------------------------------------------
# Run completion summary
# ---------------------------------------------------------------------------


def _run_completion_summary(run, aggregate: dict) -> str:
    if run.kernel == CPU_SIEVE_KERNEL:
        return (
            f"Completed odd-seed descent verification on {run.hardware} for range "
            f"{run.range_start}-{run.range_end}; max descent steps at "
            f"{aggregate['max_total_stopping_time']['n']}, max descent excursion at "
            f"{aggregate['max_excursion']['n']}."
        )
    if run.kernel == GPU_SIEVE_KERNEL:
        return (
            f"Completed odd-seed descent verification on {run.hardware} for range "
            f"{run.range_start}-{run.range_end}; max descent steps at "
            f"{aggregate['max_total_stopping_time']['n']}, max descent excursion at "
            f"{aggregate['max_excursion']['n']}."
        )
    if run.kernel == CPU_BARINA_KERNEL:
        return (
            f"Completed experimental Barina-domain descent verification on {run.hardware} "
            f"for range {run.range_start}-{run.range_end}; compressed descent record at "
            f"{aggregate['max_total_stopping_time']['n']}."
        )
    return (
        f"Completed {run.kernel} on {run.hardware} for range {run.range_start}-{run.range_end}; "
        f"max total stopping time at {aggregate['max_total_stopping_time']['n']}, "
        f"max excursion at {aggregate['max_excursion']['n']}."
    )


# ---------------------------------------------------------------------------
# Run execution
# ---------------------------------------------------------------------------


def execute_run(
    repository: LabRepository,
    run_id: str,
    *,
    checkpoint_interval: int = 250,
):
    from .services import compute_range_metrics  # lazy — orchestration → services for compute

    run = repository.get_run(run_id)
    started_at = run.started_at or utc_now()
    checkpoint = run.checkpoint or {}
    start_at = checkpoint.get("next_value", run.range_start)
    repository.update_run(run.id, status=RunStatus.RUNNING, started_at=started_at)

    aggregate = {
        "processed": 0,
        "last_processed": start_at - 1,
        "max_total_stopping_time": {"n": run.range_start, "value": -1},
        "max_stopping_time": {"n": run.range_start, "value": -1},
        "max_excursion": {"n": run.range_start, "value": -1},
        "sample_records": [],
    }

    if run.metrics:
        aggregate.update(run.metrics)

    writer = _CheckpointWriter()
    try:
        batch_start = start_at
        while batch_start <= run.range_end:
            profile = repository.get_compute_profile()
            effective_checkpoint_interval = _effective_checkpoint_interval(
                run.kernel,
                checkpoint_interval,
                profile=profile,
            )
            batch_end = min(batch_start + effective_checkpoint_interval - 1, run.range_end)
            _t0 = time.perf_counter()
            batch = compute_range_metrics(batch_start, batch_end, kernel=run.kernel, profile=profile)
            _t1 = time.perf_counter()
            aggregate["processed"] += batch.processed
            aggregate["last_processed"] = batch.last_processed
            for key in ("max_total_stopping_time", "max_stopping_time", "max_excursion"):
                if batch.__dict__[key]["value"] > aggregate[key]["value"]:
                    aggregate[key] = batch.__dict__[key]
            aggregate["sample_records"] = (
                aggregate["sample_records"] + batch.sample_records
            )[:12]

            _t2 = time.perf_counter()
            # Snapshot aggregate for the background thread (shallow copy is
            # safe because nested dicts are replaced, never mutated in-place).
            writer.submit(
                repository,
                run.id,
                checkpoint={
                    "next_value": batch_end + 1,
                    "last_processed": batch.last_processed,
                    "checkpoint_interval": effective_checkpoint_interval,
                },
                metrics=dict(aggregate),
                summary=f"Processed {aggregate['processed']} values",
            )
            _t3 = time.perf_counter()
            throttle_sec = _compute_budget_throttle_seconds(
                hardware=run.hardware,
                profile=profile,
                compute_sec=_t1 - _t0,
            )
            if throttle_sec > 0:
                time.sleep(throttle_sec)
            _t4 = time.perf_counter()
            import sys as _sys
            print(json.dumps({
                "timing": "execute_run_batch",
                "kernel": run.kernel,
                "batch_size": batch_end - batch_start + 1,
                "compute_sec": round(_t1 - _t0, 3),
                "aggregate_sec": round(_t2 - _t1, 3),
                "submit_sec": round(_t3 - _t2, 3),
                "throttle_sec": round(throttle_sec, 3),
                "total_sec": round(_t4 - _t0, 3),
                "gpu_pct": round((_t1 - _t0) / (_t4 - _t0) * 100, 1) if (_t4 - _t0) > 0 else 0,
            }), flush=True, file=_sys.stderr)
            batch_start = batch_end + 1

        # Ensure the last in-loop checkpoint is committed before completion.
        writer.shutdown()
    except Exception:
        writer.shutdown()
        raise
    finally:
        # Optional: kill stdio helper after each run to return Metal RAM (can race with overlapping
        # requests and causes stdio↔one-shot fallback warnings if another job touches Metal
        # immediately). Default off; set COLLATZ_METAL_SIEVE_STDIO_SHUTDOWN_AFTER_RUN=1 to enable.
        try:
            import os

            if os.getenv("COLLATZ_METAL_SIEVE_STDIO_SHUTDOWN_AFTER_RUN", "").strip().lower() in {
                "1",
                "true",
                "yes",
            }:
                from .gpu_sieve_metal_runtime import shutdown_metal_stdio_transport

                shutdown_metal_stdio_transport()
        except Exception:
            pass

    metrics_path = repository.settings.artifacts_dir / "runs" / f"{run.id}.json"
    metrics_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    repository.create_artifact(
        kind=ArtifactKind.JSON,
        path=metrics_path,
        run_id=run.id,
        metadata={
            "direction": run.direction_slug,
            "name": run.name,
            "kernel": run.kernel,
            "hardware": run.hardware,
        },
    )
    checksum = sha256_text(json.dumps(aggregate, sort_keys=True))
    summary = _run_completion_summary(run, aggregate)
    return repository.update_run(
        run.id,
        status=RunStatus.COMPLETED,
        checkpoint={"next_value": run.range_end + 1, "last_processed": run.range_end},
        metrics=aggregate,
        summary=summary,
        checksum=checksum,
        finished_at=utc_now(),
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(repository: LabRepository) -> Path:
    summary = repository.summary()
    directions = repository.list_directions()
    runs = repository.list_runs()[:10]
    claims = repository.list_claims()[:10]
    report_path = repository.settings.reports_dir / f"lab-report-{utc_now().replace(':', '-')}.md"

    lines = [
        "# Collatz Lab Report",
        "",
        f"- Directions: {summary.direction_count}",
        f"- Runs: {summary.run_count}",
        f"- Validated runs: {summary.validated_run_count}",
        f"- Queued runs: {summary.queued_run_count}",
        f"- Running runs: {summary.running_run_count}",
        f"- Claims: {summary.claim_count}",
        f"- Open tasks: {summary.open_task_count}",
        f"- Artifacts: {summary.artifact_count}",
        f"- Workers: {summary.worker_count}",
        "",
        "## Directions",
        "",
    ]

    for direction in directions:
        lines.extend(
            [
                f"### {direction.title}",
                "",
                f"- Slug: `{direction.slug}`",
                f"- Status: `{direction.status}`",
                f"- Score: `{direction.score}`",
                f"- Success: {direction.success_criteria}",
                f"- Abandon: {direction.abandon_criteria}",
                "",
            ]
        )

    lines.extend(["## Recent Runs", ""])
    for run in runs:
        lines.extend(
            [
                f"- `{run.id}` {run.name}: `{run.status}` on [{run.range_start}, {run.range_end}]",
                f"  Summary: {run.summary}",
            ]
        )

    lines.extend(["", "## Recent Claims", ""])
    for claim in claims:
        lines.extend(
            [
                f"- `{claim.id}` {claim.title}: `{claim.status}`",
                f"  Statement: {claim.statement}",
            ]
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    repository.create_artifact(
        kind=ArtifactKind.REPORT,
        path=report_path,
        metadata={"generated_at": utc_now(), "type": "lab-report"},
    )
    return report_path


# ---------------------------------------------------------------------------
# Worker queue dispatch
# ---------------------------------------------------------------------------


def process_next_queued_run(
    repository: LabRepository,
    *,
    worker_id: str,
    supported_hardware: list[str],
    supported_kernels: list[str],
    checkpoint_interval: int = 250,
    validate_after_run: bool = False,
):
    _ensure_overflow_recovery_runs(repository)
    annotate_legacy_validation_failures(repository)
    run = repository.claim_next_run(
        worker_id=worker_id,
        supported_hardware=supported_hardware,
        supported_kernels=supported_kernels,
    )
    if run is None:
        with _serialized_maintenance_enqueue(repository):
            run = repository.claim_next_run(
                worker_id=worker_id,
                supported_hardware=supported_hardware,
                supported_kernels=supported_kernels,
            )
            if run is None:
                queued_ids = queue_legacy_validation_reruns(
                    repository,
                    supported_hardware=supported_hardware,
                    limit=4 if "cpu" in supported_hardware else 2,
                )
                if not queued_ids:
                    queued_ids = queue_continuous_verification_runs(
                        repository,
                        supported_hardware=supported_hardware,
                    )
                if not queued_ids:
                    queued_ids = queue_research_snack_runs(
                        repository,
                        supported_hardware=supported_hardware,
                        supported_kernels=supported_kernels,
                    )
                if not queued_ids:
                    queued_ids = queue_randomized_compute_runs(
                        repository,
                        supported_hardware=supported_hardware,
                        supported_kernels=supported_kernels,
                    )
                if not queued_ids:
                    return None
                run = repository.claim_next_run(
                    worker_id=worker_id,
                    supported_hardware=supported_hardware,
                    supported_kernels=supported_kernels,
                )
        if run is None:
            return None

    try:
        completed_run = execute_run(
            repository,
            run.id,
            checkpoint_interval=checkpoint_interval,
        )
        if validate_after_run:
            from .validation import validate_run  # lazy — validation imports from services
            completed_run = validate_run(repository, run.id)
        repository.update_worker(worker_id, status="idle", current_run_id=None)
        return completed_run
    except Exception as exc:
        tb_text = _traceback_mod.format_exc()
        logger.error(
            "Run %s failed on worker %s:\n%s", run.id, worker_id, tb_text,
        )
        # Store traceback tail in summary for post-mortem analysis
        short_tb = tb_text[-400:] if len(tb_text) > 400 else tb_text
        failed_run = repository.update_run(
            run.id,
            status=RunStatus.FAILED,
            summary=f"Worker {worker_id} failed: {exc}\n---\n{short_tb}",
            finished_at=utc_now(),
        )
        recovery_ids = _ensure_overflow_recovery_runs(repository, run_ids={run.id})
        if recovery_ids:
            failed_run = repository.update_run(
                run.id,
                summary=(
                    f"Worker {worker_id} failed: {exc} "
                    f"Recovery queued via {', '.join(recovery_ids)}.\n---\n{short_tb}"
                ),
            )
        repository.update_worker(worker_id, status="idle", current_run_id=None)
        return failed_run


# ---------------------------------------------------------------------------
# Modular claim probing
# ---------------------------------------------------------------------------


def probe_modular_claim(
    *,
    modulus: int,
    allowed_residues: list[int],
    search_limit: int,
) -> ModularProbeResult:
    if modulus < 2:
        raise ValueError("Modulus must be at least 2.")
    if search_limit < 3:
        raise ValueError("Search limit must be at least 3.")

    normalized_residues = sorted({int(value) % modulus for value in allowed_residues})
    checked_odd_values = 0
    counterexamples: list[int] = []

    # Safety bound: the longest known Collatz sequence for seeds below 10^9
    # converges within ~1000 steps.  We use a generous limit that scales with
    # seed magnitude so we never silently skip a seed that actually converges.
    max_steps = max(MAX_KERNEL_STEPS, search_limit * 10)

    skipped_seeds: list[int] = []
    for start in range(3, search_limit + 1, 2):
        checked_odd_values += 1
        current = start
        steps = 0
        while current != 1 and steps < max_steps:
            current = collatz_step(current)
            steps += 1
        if current != 1:
            skipped_seeds.append(start)
            continue
        if start % modulus not in normalized_residues:
            counterexamples.append(start)
            if len(counterexamples) >= 12:
                break

    first_counterexample = counterexamples[0] if counterexamples else None
    if first_counterexample is None:
        rationale = (
            f"No counterexample was found among odd seeds up to {search_limit} for the residue rule "
            f"mod {modulus} in {normalized_residues}."
        )
    else:
        rationale = (
            f"Found counterexample {first_counterexample}: it reaches 1 in the checked range but its residue "
            f"{first_counterexample % modulus} is outside {normalized_residues} modulo {modulus}."
        )
    if skipped_seeds:
        rationale += (
            f" WARNING: {len(skipped_seeds)} seed(s) did not converge within {max_steps} steps "
            f"and were excluded from analysis (first: {skipped_seeds[0]}). "
            f"These may represent extremely long orbits or potential non-convergence."
        )

    return ModularProbeResult(
        modulus=modulus,
        allowed_residues=normalized_residues,
        checked_limit=search_limit,
        checked_odd_values=checked_odd_values,
        first_counterexample=first_counterexample,
        counterexamples=counterexamples,
        rationale=rationale,
    )
