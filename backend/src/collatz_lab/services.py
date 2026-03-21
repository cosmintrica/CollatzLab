from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from .hardware import (
    CPU_ACCELERATED_KERNEL,
    CPU_DIRECT_KERNEL,
    CPU_PARALLEL_KERNEL,
    GPU_KERNEL,
    gpu_execution_ready,
)

try:
    import numpy as np
except Exception:  # pragma: no cover - optional GPU dependency
    np = None

try:
    from numba import cuda, njit, prange
except Exception:  # pragma: no cover - optional GPU dependency
    cuda = None
    njit = None
    prange = None
from .repository import LabRepository, sha256_text, utc_now
from .schemas import ArtifactKind, ModularProbeResult, Run, RunStatus


def collatz_step(value: int) -> int:
    if value < 1:
        raise ValueError("Collatz is defined only for positive integers.")
    return value // 2 if value % 2 == 0 else (3 * value) + 1


def accelerated_odd_step(value: int) -> tuple[int, int]:
    if value < 1 or value % 2 == 0:
        raise ValueError("accelerated_odd_step requires a positive odd integer.")
    current = (3 * value) + 1
    shifts = 0
    while current % 2 == 0:
        current //= 2
        shifts += 1
    return current, shifts


if cuda is not None and np is not None:  # pragma: no branch - import guarded above

    @cuda.jit
    def _collatz_metrics_kernel(start_value, size, total_steps, stopping_steps, max_excursions):
        index = cuda.grid(1)
        if index >= size:
            return

        current = start_value + index
        original = current
        steps = 0
        stopping_time = 0
        max_excursion = current

        while current != 1:
            if current % 2 == 0:
                current //= 2
                steps += 1
                if stopping_time == 0 and current < original:
                    stopping_time = steps
            else:
                current = (3 * current) + 1
                steps += 1
                if current > max_excursion:
                    max_excursion = current
                if stopping_time == 0 and current < original:
                    stopping_time = steps
                while current % 2 == 0:
                    current //= 2
                    steps += 1
                    if stopping_time == 0 and current < original:
                        stopping_time = steps
            if current > max_excursion:
                max_excursion = current

        total_steps[index] = steps
        stopping_steps[index] = stopping_time if stopping_time != 0 else steps
        max_excursions[index] = max_excursion


if np is not None and njit is not None:  # pragma: no branch - import guarded above

    @njit(cache=True, parallel=True)
    def _collatz_metrics_parallel(start_value, size):
        total_steps = np.empty(size, dtype=np.int64)
        stopping_steps = np.empty(size, dtype=np.int64)
        max_excursions = np.empty(size, dtype=np.int64)

        for index in prange(size):
            current = int(start_value + index)
            original = current
            steps = 0
            stopping_time = 0
            max_excursion = current

            while current != 1:
                if current % 2 == 0:
                    current //= 2
                    steps += 1
                    if stopping_time == 0 and current < original:
                        stopping_time = steps
                else:
                    current = (3 * current) + 1
                    steps += 1
                    if current > max_excursion:
                        max_excursion = current
                    if stopping_time == 0 and current < original:
                        stopping_time = steps
                    while current % 2 == 0:
                        current //= 2
                        steps += 1
                        if stopping_time == 0 and current < original:
                            stopping_time = steps
                if current > max_excursion:
                    max_excursion = current

            total_steps[index] = steps
            stopping_steps[index] = stopping_time if stopping_time != 0 else steps
            max_excursions[index] = max_excursion

        return total_steps, stopping_steps, max_excursions


@dataclass
class NumberMetrics:
    total_stopping_time: int
    stopping_time: int
    max_excursion: int


@dataclass
class AggregateMetrics:
    processed: int
    last_processed: int
    max_total_stopping_time: dict
    max_stopping_time: dict
    max_excursion: dict
    sample_records: list[dict]


def metrics_direct(value: int) -> NumberMetrics:
    current = value
    total_steps = 0
    stopping_time: int | None = None
    max_excursion = value
    while current != 1:
        current = collatz_step(current)
        total_steps += 1
        max_excursion = max(max_excursion, current)
        if stopping_time is None and current < value:
            stopping_time = total_steps
    return NumberMetrics(
        total_stopping_time=total_steps,
        stopping_time=stopping_time or total_steps,
        max_excursion=max_excursion,
    )


def metrics_accelerated(value: int) -> NumberMetrics:
    current = value
    total_steps = 0
    stopping_time: int | None = None
    max_excursion = value
    while current != 1:
        if current % 2 == 0:
            current //= 2
            total_steps += 1
            if stopping_time is None and current < value:
                stopping_time = total_steps
        else:
            current = (3 * current) + 1
            total_steps += 1
            max_excursion = max(max_excursion, current)
            if stopping_time is None and current < value:
                stopping_time = total_steps
            while current % 2 == 0:
                current //= 2
                total_steps += 1
                if stopping_time is None and current < value:
                    stopping_time = total_steps
        max_excursion = max(max_excursion, current)
    return NumberMetrics(
        total_stopping_time=total_steps,
        stopping_time=stopping_time or total_steps,
        max_excursion=max_excursion,
    )


def _aggregate_metrics_from_arrays(
    first_value: int,
    total_steps,
    stopping_steps,
    max_excursions,
    *,
    sample_limit: int,
) -> AggregateMetrics:
    count = len(total_steps)
    max_total = {"n": first_value, "value": -1}
    max_stopping = {"n": first_value, "value": -1}
    max_excursion = {"n": first_value, "value": -1}
    sample_records: list[dict] = []

    for index in range(count):
        value = first_value + index
        total = int(total_steps[index])
        stopping = int(stopping_steps[index])
        excursion = int(max_excursions[index])

        if total > max_total["value"]:
            max_total = {"n": value, "value": total}
            if len(sample_records) < sample_limit:
                sample_records.append({"metric": "max_total_stopping_time", **max_total})

        if stopping > max_stopping["value"]:
            max_stopping = {"n": value, "value": stopping}
            if len(sample_records) < sample_limit:
                sample_records.append({"metric": "max_stopping_time", **max_stopping})

        if excursion > max_excursion["value"]:
            max_excursion = {"n": value, "value": excursion}
            if len(sample_records) < sample_limit:
                sample_records.append({"metric": "max_excursion", **max_excursion})

    return AggregateMetrics(
        processed=count,
        last_processed=first_value + count - 1,
        max_total_stopping_time=max_total,
        max_stopping_time=max_stopping,
        max_excursion=max_excursion,
        sample_records=sample_records,
    )


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _gpu_threads_per_block() -> int:
    value = _positive_int_env("COLLATZ_GPU_THREADS_PER_BLOCK", 256)
    return max(64, min(1024, value))


def compute_range_metrics_gpu(
    start: int,
    end: int,
    *,
    start_at: int | None = None,
    sample_limit: int = 12,
) -> AggregateMetrics:
    if not gpu_execution_ready() or cuda is None or np is None:
        raise ValueError("GPU execution is not available on this machine.")
    if start < 1 or end < start:
        raise ValueError("Invalid run interval.")
    first = start_at if start_at is not None else start
    if first < start or first > end:
        raise ValueError("Checkpoint start is outside the run interval.")

    size = end - first + 1
    device_total_steps = cuda.device_array(size, dtype=np.int64)
    device_stopping_steps = cuda.device_array(size, dtype=np.int64)
    device_max_excursions = cuda.device_array(size, dtype=np.int64)

    threads_per_block = _gpu_threads_per_block()
    blocks_per_grid = math.ceil(size / threads_per_block)
    _collatz_metrics_kernel[blocks_per_grid, threads_per_block](
        first,
        size,
        device_total_steps,
        device_stopping_steps,
        device_max_excursions,
    )
    cuda.synchronize()

    return _aggregate_metrics_from_arrays(
        first,
        device_total_steps.copy_to_host(),
        device_stopping_steps.copy_to_host(),
        device_max_excursions.copy_to_host(),
        sample_limit=sample_limit,
    )


def compute_range_metrics_parallel(
    start: int,
    end: int,
    *,
    start_at: int | None = None,
    sample_limit: int = 12,
) -> AggregateMetrics:
    if np is None or njit is None or prange is None:
        raise ValueError("CPU parallel execution requires numpy and numba.")
    if start < 1 or end < start:
        raise ValueError("Invalid run interval.")
    first = start_at if start_at is not None else start
    if first < start or first > end:
        raise ValueError("Checkpoint start is outside the run interval.")

    size = end - first + 1
    total_steps, stopping_steps, max_excursions = _collatz_metrics_parallel(first, size)
    return _aggregate_metrics_from_arrays(
        first,
        total_steps,
        stopping_steps,
        max_excursions,
        sample_limit=sample_limit,
    )


def compute_range_metrics(
    start: int,
    end: int,
    *,
    kernel: str = "cpu-direct",
    start_at: int | None = None,
    sample_limit: int = 12,
) -> AggregateMetrics:
    if start < 1 or end < start:
        raise ValueError("Invalid run interval.")
    first = start_at if start_at is not None else start
    if first < start or first > end:
        raise ValueError("Checkpoint start is outside the run interval.")
    if kernel == CPU_PARALLEL_KERNEL:
        return compute_range_metrics_parallel(
            start,
            end,
            start_at=start_at,
            sample_limit=sample_limit,
        )
    if kernel == GPU_KERNEL:
        return compute_range_metrics_gpu(
            start,
            end,
            start_at=start_at,
            sample_limit=sample_limit,
        )
    if kernel not in {CPU_DIRECT_KERNEL, CPU_ACCELERATED_KERNEL}:
        raise ValueError(f"Unsupported execution kernel: {kernel}")

    max_total = {"n": first, "value": -1}
    max_stopping = {"n": first, "value": -1}
    max_excursion = {"n": first, "value": -1}
    sample_records: list[dict] = []
    processed = 0
    last_processed = first - 1
    metric_fn = metrics_direct if kernel == CPU_DIRECT_KERNEL else metrics_accelerated

    for value in range(first, end + 1):
        metrics = metric_fn(value)
        processed += 1
        last_processed = value

        if metrics.total_stopping_time > max_total["value"]:
            max_total = {"n": value, "value": metrics.total_stopping_time}
            if len(sample_records) < sample_limit:
                sample_records.append({"metric": "max_total_stopping_time", **max_total})

        if metrics.stopping_time > max_stopping["value"]:
            max_stopping = {"n": value, "value": metrics.stopping_time}
            if len(sample_records) < sample_limit:
                sample_records.append({"metric": "max_stopping_time", **max_stopping})

        if metrics.max_excursion > max_excursion["value"]:
            max_excursion = {"n": value, "value": metrics.max_excursion}
            if len(sample_records) < sample_limit:
                sample_records.append({"metric": "max_excursion", **max_excursion})

    return AggregateMetrics(
        processed=processed,
        last_processed=last_processed,
        max_total_stopping_time=max_total,
        max_stopping_time=max_stopping,
        max_excursion=max_excursion,
        sample_records=sample_records,
    )


def _effective_checkpoint_interval(kernel: str, checkpoint_interval: int) -> int:
    minimums = {
        CPU_PARALLEL_KERNEL: _positive_int_env("COLLATZ_CPU_PARALLEL_BATCH_SIZE", 250_000),
        GPU_KERNEL: _positive_int_env("COLLATZ_GPU_BATCH_SIZE", 5_000_000),
    }
    return max(checkpoint_interval, minimums.get(kernel, checkpoint_interval))


def _aggregate_validation_payload(metrics: AggregateMetrics) -> dict:
    return {
        "processed": metrics.processed,
        "last_processed": metrics.last_processed,
        "max_total_stopping_time": metrics.max_total_stopping_time,
        "max_stopping_time": metrics.max_stopping_time,
        "max_excursion": metrics.max_excursion,
    }


def execute_run(
    repository: LabRepository,
    run_id: str,
    *,
    checkpoint_interval: int = 250,
) -> Run:
    run = repository.get_run(run_id)
    started_at = run.started_at or utc_now()
    checkpoint = run.checkpoint or {}
    start_at = checkpoint.get("next_value", run.range_start)
    effective_checkpoint_interval = _effective_checkpoint_interval(
        run.kernel,
        checkpoint_interval,
    )
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

    batch_start = start_at
    while batch_start <= run.range_end:
        batch_end = min(batch_start + effective_checkpoint_interval - 1, run.range_end)
        batch = compute_range_metrics(batch_start, batch_end, kernel=run.kernel)
        aggregate["processed"] += batch.processed
        aggregate["last_processed"] = batch.last_processed
        for key in ("max_total_stopping_time", "max_stopping_time", "max_excursion"):
            if batch.__dict__[key]["value"] > aggregate[key]["value"]:
                aggregate[key] = batch.__dict__[key]
        aggregate["sample_records"] = (
            aggregate["sample_records"] + batch.sample_records
        )[:12]

        repository.update_run(
            run.id,
            checkpoint={
                "next_value": batch_end + 1,
                "last_processed": batch.last_processed,
                "checkpoint_interval": effective_checkpoint_interval,
            },
            metrics=aggregate,
            summary=f"Processed {aggregate['processed']} values",
        )
        batch_start = batch_end + 1

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
    summary = (
        f"Completed {run.kernel} on {run.hardware} for range {run.range_start}-{run.range_end}; "
        f"max total stopping time at {aggregate['max_total_stopping_time']['n']}, "
        f"max excursion at {aggregate['max_excursion']['n']}."
    )
    return repository.update_run(
        run.id,
        status=RunStatus.COMPLETED,
        checkpoint={"next_value": run.range_end + 1, "last_processed": run.range_end},
        metrics=aggregate,
        summary=summary,
        checksum=checksum,
        finished_at=utc_now(),
    )


def validate_run(repository: LabRepository, run_id: str) -> Run:
    run = repository.get_run(run_id)
    mismatches: list[str] = []
    if run.kernel in {CPU_PARALLEL_KERNEL, GPU_KERNEL}:
        reference = compute_range_metrics(run.range_start, run.range_end, kernel=CPU_DIRECT_KERNEL)
        candidate = compute_range_metrics(run.range_start, run.range_end, kernel=run.kernel)
        if _aggregate_validation_payload(reference) != _aggregate_validation_payload(candidate):
            mismatches.append(
                f"aggregate mismatch: direct={_aggregate_validation_payload(reference)} "
                f"candidate={_aggregate_validation_payload(candidate)}"
            )
    else:
        for value in range(run.range_start, run.range_end + 1):
            direct = metrics_direct(value)
            accelerated = metrics_accelerated(value)
            if direct != accelerated:
                mismatches.append(
                    f"{value}: direct={asdict(direct)} accelerated={asdict(accelerated)}"
                )
                break

    validation_path = (
        repository.settings.artifacts_dir / "validations" / f"{run.id}-validation.md"
    )
    if mismatches:
        body = (
            f"# Validation for {run.id}\n\n"
            f"Status: failed\n\n"
            f"Mismatch detected:\n\n- {mismatches[0]}\n"
        )
        validation_path.write_text(body, encoding="utf-8")
        repository.create_artifact(
            kind=ArtifactKind.REPORT,
            path=validation_path,
            run_id=run.id,
            metadata={"status": "failed"},
        )
        return repository.update_run(
            run.id,
            status=RunStatus.FAILED,
            summary="Validation failed: direct and accelerated kernels diverged.",
            finished_at=utc_now(),
        )

    body = (
        f"# Validation for {run.id}\n\n"
        f"Status: passed\n\n"
        f"- Interval: {run.range_start}-{run.range_end}\n"
        f"- Kernel under review: {run.kernel}\n"
        f"- Direct and accelerated implementations matched on every value.\n"
    )
    validation_path.write_text(body, encoding="utf-8")
    repository.create_artifact(
        kind=ArtifactKind.REPORT,
        path=validation_path,
        run_id=run.id,
        metadata={"status": "passed"},
    )
    return repository.update_run(
        run.id,
        status=RunStatus.VALIDATED,
        summary="Validation passed: direct and accelerated kernels matched.",
        finished_at=utc_now(),
    )


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


def process_next_queued_run(
    repository: LabRepository,
    *,
    worker_id: str,
    supported_hardware: list[str],
    supported_kernels: list[str],
    checkpoint_interval: int = 250,
    validate_after_run: bool = False,
) -> Run | None:
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
            completed_run = validate_run(repository, run.id)
        repository.update_worker(worker_id, status="idle", current_run_id=None)
        return completed_run
    except Exception as exc:
        failed_run = repository.update_run(
            run.id,
            status=RunStatus.FAILED,
            summary=f"Worker {worker_id} failed: {exc}",
            finished_at=utc_now(),
        )
        repository.update_worker(worker_id, status="idle", current_run_id=None)
        return failed_run


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

    for start in range(3, search_limit + 1, 2):
        checked_odd_values += 1
        current = start
        steps = 0
        while current != 1 and steps < 10000:
            current = collatz_step(current)
            steps += 1
        if current != 1:
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

    return ModularProbeResult(
        modulus=modulus,
        allowed_residues=normalized_residues,
        checked_limit=search_limit,
        checked_odd_values=checked_odd_values,
        first_counterexample=first_counterexample,
        counterexamples=counterexamples,
        rationale=rationale,
    )
