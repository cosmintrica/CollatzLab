"""Run scheduling, autopilot, overflow recovery, and legacy revalidation for Collatz Lab.

Extracted from ``services.py``.  Has no imports from ``services`` or
``orchestration`` — the dependency graph is:

    metrics_sot / schemas / repository / hardware
        ↑
    _profile_helpers
        ↑
    scheduling          ← (this module)
        ↑
    orchestration
        ↑
    services  (re-exports for backward-compat)
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import random as _random

from datetime import datetime

from ._profile_helpers import _effective_profile_percent
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
from .metrics_sot import INT64_MAX
from .repository import LabRepository, sha256_text, utc_now
from .schemas import ComputeProfile, Run, RunStatus

logger = logging.getLogger("collatz_lab.scheduling")

# ---------------------------------------------------------------------------
# Owner / prefix constants
# ---------------------------------------------------------------------------

OVERFLOW_RECOVERY_OWNER = "overflow-recovery"
RESEARCH_AUTOPILOT_OWNER = "collatz-research-autopilot"
RANDOM_PROBE_OWNER = "collatz-random-probe"
LEGACY_VALIDATION_RERUN_OWNER = "legacy-revalidation"
LEGACY_VALIDATION_RERUN_PREFIX = "legacy-rerun-"


# Full-orbit / heavy-per-seed kernels: autopilot & probes must not pick intervals
# starting near 10^10+ — wall time explodes vs sieve-style kernels.
ORBIT_STYLE_PROBE_MAX_RANGE_END: int = 8_000_000

_ORBIT_BOUNDED_BASE_KERNELS: frozenset[str] = frozenset(
    {
        CPU_DIRECT_KERNEL,
        CPU_ACCELERATED_KERNEL,
        CPU_PARALLEL_KERNEL,
        CPU_PARALLEL_ODD_KERNEL,
        GPU_KERNEL,
    }
)


# ---------------------------------------------------------------------------
# Process-level serialisation lock for maintenance enqueue
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _serialized_maintenance_enqueue(repository: LabRepository):
    """Serialize autopilot / legacy / snack enqueue across worker *processes*.

    If two workers poll with an empty queue, both could call
    ``queue_continuous_verification_runs`` before either marks a GPU run RUNNING,
    see ``active_gpu == False`` twice, and insert **two** ``autopilot-continuous-gpu`` rows.
    """
    lock_path = repository.settings.db_path.resolve().parent / ".collatz_maintenance_enqueue.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        try:
            from filelock import FileLock
        except ImportError:
            logger.warning(
                "filelock not installed: on Windows, install `filelock` to avoid duplicate autopilot enqueue."
            )
            yield
            return
        with FileLock(str(lock_path), timeout=600):
            yield
        return

    import fcntl

    fp = open(lock_path, "a+b")
    try:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
        fp.close()


# ---------------------------------------------------------------------------
# Randomness helpers
# ---------------------------------------------------------------------------


def _lab_random() -> _random.Random:
    """Deterministic runs in tests via ``COLLATZ_RANDOM_SEED`` (int or string)."""
    raw = os.getenv("COLLATZ_RANDOM_SEED", "").strip()
    if raw:
        try:
            return _random.Random(int(raw))
        except ValueError:
            return _random.Random(raw)
    return _random.Random()


def choose_probe_interval_start(
    kernel: str,
    span: int,
    rng: _random.Random,
    *,
    loose_upper_base: int,
) -> int | None:
    """Pick ``range_start`` for a random probe/snack.

    Orbit-style kernels (direct, accelerated, parallel, GPU orbit) are capped so
    ``range_end <= ORBIT_STYLE_PROBE_MAX_RANGE_END``. Sieve kernels may use
    ``loose_upper_base`` (billions) as before.
    """
    if kernel in _ORBIT_BOUNDED_BASE_KERNELS:
        hi = ORBIT_STYLE_PROBE_MAX_RANGE_END - span
        if hi < 1:
            return None
        return rng.randint(1, hi)
    return rng.randint(1, loose_upper_base)


def _random_probes_enabled() -> bool:
    return os.getenv("COLLATZ_RANDOM_PROBES", "1").strip().lower() not in ("0", "false", "no", "off")


# ---------------------------------------------------------------------------
# Overflow-guard helpers
# ---------------------------------------------------------------------------


def _is_overflow_guard_failure(run: Run) -> bool:
    return (
        run.direction_slug == "verification"
        and run.status == RunStatus.FAILED
        and "overflow guard triggered" in (run.summary or "").lower()
    )


def _prune_duplicate_overflow_recovery_runs(repository: LabRepository) -> list[str]:
    queued_duplicates: dict[tuple[str, int, int, str, str, str], list[Run]] = {}
    for run in repository.list_runs():
        if run.owner != OVERFLOW_RECOVERY_OWNER or not run.name.startswith("recover-"):
            continue
        key = (run.name, run.range_start, run.range_end, run.hardware, run.kernel, run.owner)
        queued_duplicates.setdefault(key, []).append(run)

    removed_ids: list[str] = []
    for duplicates in queued_duplicates.values():
        if len(duplicates) < 2:
            continue
        duplicates = sorted(duplicates, key=lambda item: (item.created_at, item.id))
        keep = duplicates[0]
        for duplicate in duplicates[1:]:
            # Only delete synthetic duplicates that never started; preserve real work.
            if duplicate.status == RunStatus.QUEUED:
                repository.delete_run(duplicate.id)
                removed_ids.append(duplicate.id)
            elif keep.status == RunStatus.QUEUED and duplicate.status in {
                RunStatus.RUNNING,
                RunStatus.COMPLETED,
                RunStatus.VALIDATED,
            }:
                repository.delete_run(keep.id)
                removed_ids.append(keep.id)
                keep = duplicate
    return removed_ids


def _ensure_overflow_recovery_runs(
    repository: LabRepository,
    *,
    run_ids: set[str] | None = None,
) -> list[str]:
    _prune_duplicate_overflow_recovery_runs(repository)
    runs = repository.list_runs()
    runs_by_name = {run.name: run for run in runs}
    created_ids: list[str] = []

    for failed_run in sorted(runs, key=lambda item: (item.created_at, item.id)):
        if not _is_overflow_guard_failure(failed_run):
            continue
        if run_ids is not None and failed_run.id not in run_ids:
            continue

        current = repository.get_run(failed_run.id)
        checkpoint = current.checkpoint or {}
        metrics = current.metrics or {}
        last_processed = int(
            checkpoint.get("last_processed")
            or metrics.get("last_processed")
            or (current.range_start - 1)
        )

        if last_processed >= current.range_start:
            prefix_name = f"recover-prefix-{current.id}"
            if prefix_name not in runs_by_name:
                prefix_run = repository.create_run(
                    direction_slug=current.direction_slug,
                    name=prefix_name,
                    range_start=current.range_start,
                    range_end=last_processed,
                    kernel=current.kernel,
                    owner=OVERFLOW_RECOVERY_OWNER,
                    hardware=current.hardware,
                )
                checksum = (
                    sha256_text(json.dumps(metrics, sort_keys=True))
                    if metrics
                    else prefix_run.checksum
                )
                prefix_run = repository.update_run(
                    prefix_run.id,
                    status=RunStatus.COMPLETED,
                    checkpoint={
                        "next_value": last_processed + 1,
                        "last_processed": last_processed,
                    },
                    metrics=metrics,
                    summary=(
                        f"Recovered exact prefix from {current.id} after overflow guard "
                        f"stopped the original run. Covers {current.range_start}-{last_processed}."
                    ),
                    checksum=checksum,
                    started_at=current.started_at or prefix_run.created_at,
                    finished_at=utc_now(),
                )
                runs_by_name[prefix_name] = prefix_run
                created_ids.append(prefix_run.id)

        recovery_start = max(current.range_start, last_processed + 1)
        if recovery_start <= current.range_end:
            tail_name = f"recover-tail-{current.id}"
            if tail_name not in runs_by_name:
                recovery_run = repository.create_run(
                    direction_slug=current.direction_slug,
                    name=tail_name,
                    range_start=recovery_start,
                    range_end=current.range_end,
                    kernel=CPU_PARALLEL_KERNEL,
                    owner=OVERFLOW_RECOVERY_OWNER,
                    hardware="cpu",
                )
                recovery_run = repository.update_run(
                    recovery_run.id,
                    summary=(
                        f"Exact CPU recovery queued for the uncovered tail after {current.id} "
                        f"hit the signed-64-bit overflow frontier on {current.hardware}. "
                        f"Covers {recovery_start}-{current.range_end}."
                    ),
                )
                runs_by_name[tail_name] = recovery_run
                created_ids.append(recovery_run.id)

    return created_ids


# ---------------------------------------------------------------------------
# Autopilot: continuous verification
# ---------------------------------------------------------------------------


def queue_continuous_verification_runs(
    repository: LabRepository,
    *,
    supported_hardware: list[str] | None = None,
    owner: str = "gemini-autopilot",
) -> list[str]:
    _ensure_overflow_recovery_runs(repository)
    runs = repository.list_runs()
    allowed_hardware = set(supported_hardware or ["cpu", "gpu"])
    overflow_history_hardware = {
        run.hardware
        for run in runs
        if _is_overflow_guard_failure(run)
    }
    profile = repository.get_compute_profile()
    if not profile.continuous_enabled:
        return []
    active_runs = [
        run
        for run in runs
        if run.status.value in {"queued", "running"}
    ]
    active_cpu = any(run.hardware == "cpu" for run in active_runs)
    active_gpu = any(run.hardware == "gpu" for run in active_runs)
    covered_runs = [
        run
        for run in runs
        if run.status in {RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.COMPLETED, RunStatus.VALIDATED}
    ]
    next_start = max((run.range_end for run in covered_runs), default=0) + 1
    queued_ids: list[str] = []

    cpu_intensity = _effective_profile_percent(profile, "cpu")
    gpu_intensity = _effective_profile_percent(profile, "gpu")
    # Dynamic span: look at the 3 most recent completed CPU sieve runs and scale up if fast.
    # ``list_runs`` is ordered ``created_at DESC`` (newest first), so we want ``[:3]``, not ``[-3:]``
    # (which incorrectly took the three *oldest* matches and stalled span tuning).
    recent_cpu_runs = [
        r for r in runs
        if r.hardware == "cpu"
        and r.kernel == CPU_SIEVE_KERNEL
        and r.status in {RunStatus.COMPLETED, RunStatus.VALIDATED}
        and r.started_at
        and r.finished_at
    ][:3]

    base_cpu_span = 2_000_000_000
    if len(recent_cpu_runs) >= 2:
        durations = []
        for r in recent_cpu_runs:
            try:
                start = datetime.fromisoformat(r.started_at.replace("Z", "+00:00"))
                finish = datetime.fromisoformat(r.finished_at.replace("Z", "+00:00"))
                durations.append((finish - start).total_seconds())
            except Exception:
                pass
        if durations:
            avg_secs = sum(durations) / len(durations)
            run_size = recent_cpu_runs[0].range_end - recent_cpu_runs[0].range_start + 1
            if avg_secs > 0 and run_size > 0:
                # Target: runs should take ~120 seconds for good checkpoint cadence
                # No upper cap — span grows freely until runs naturally reach ~120s
                target_span = int(run_size * 120 / max(avg_secs, 5))
                base_cpu_span = max(500_000_000, target_span)

    # Sieve kernel uses standard Collatz steps with early termination.
    # Every odd seed is individually verified — no seeds are skipped.
    cpu_span = max(100_000_000, round(base_cpu_span * max(cpu_intensity, 5) / 100))

    # Dynamic GPU span: same strategy as CPU — target ~120s per run
    # GPU sieve is full-range Numba-bound, so it's slower than CPU per seed.
    recent_gpu_runs = [
        r for r in runs
        if r.hardware == "gpu"
        and r.kernel == GPU_SIEVE_KERNEL
        and r.status in {RunStatus.COMPLETED, RunStatus.VALIDATED}
        and r.started_at
        and r.finished_at
    ][:3]
    base_gpu_span = 1_000_000_000
    if len(recent_gpu_runs) >= 2:
        gpu_durations = []
        for r in recent_gpu_runs:
            try:
                start = datetime.fromisoformat(r.started_at.replace("Z", "+00:00"))
                finish = datetime.fromisoformat(r.finished_at.replace("Z", "+00:00"))
                gpu_durations.append((finish - start).total_seconds())
            except Exception:
                pass
        if gpu_durations:
            avg_gpu_secs = sum(gpu_durations) / len(gpu_durations)
            gpu_run_size = recent_gpu_runs[0].range_end - recent_gpu_runs[0].range_start + 1
            if avg_gpu_secs > 0 and gpu_run_size > 0:
                # No upper cap — span grows freely until runs naturally reach ~120s
                target_gpu_span = int(gpu_run_size * 120 / max(avg_gpu_secs, 10))
                base_gpu_span = max(250_000_000, target_gpu_span)
    gpu_span = max(250_000_000, round(base_gpu_span * max(gpu_intensity, 5) / 100))

    if "cpu" in allowed_hardware and cpu_intensity > 0 and not active_cpu:
        cpu_end = next_start + cpu_span - 1
        run = repository.create_run(
            direction_slug="verification",
            name="autopilot-continuous-cpu",
            range_start=next_start,
            range_end=cpu_end,
            kernel=CPU_SIEVE_KERNEL,
            hardware="cpu",
            owner=owner,
        )
        queued_ids.append(run.id)
        next_start = cpu_end + 1

    if "gpu" in allowed_hardware and gpu_intensity > 0 and not active_gpu:
        gpu_end = next_start + gpu_span - 1
        run = repository.create_run(
            direction_slug="verification",
            name="autopilot-continuous-gpu",
            range_start=next_start,
            range_end=gpu_end,
            kernel=GPU_SIEVE_KERNEL,
            hardware="gpu",
            owner=owner,
        )
        queued_ids.append(run.id)

    return queued_ids


# ---------------------------------------------------------------------------
# Autopilot: research snacks
# ---------------------------------------------------------------------------


def queue_research_snack_runs(
    repository: LabRepository,
    *,
    supported_hardware: list[str] | None = None,
    supported_kernels: list[str] | None = None,
) -> list[str]:
    """When the queue is still empty after autopilot, enqueue a small CPU research run.

    Picks a **random** supported kernel from the same pool and a **random** compact interval
    (seed may be fixed with ``COLLATZ_RANDOM_SEED`` for tests).

    Bounded: at most one queued research snack at a time; respects ``continuous_enabled``
    and CPU lane budget (>0).
    """
    profile = repository.get_compute_profile()
    if not profile.continuous_enabled:
        return []
    if _effective_profile_percent(profile, "cpu") <= 0:
        return []
    allowed_hw = set(supported_hardware or ["cpu", "gpu"])
    if "cpu" not in allowed_hw:
        return []
    kernels = list(supported_kernels or [])
    preferred = [
        CPU_BARINA_KERNEL,
        CPU_SIEVE_KERNEL,
        CPU_PARALLEL_KERNEL,
        CPU_ACCELERATED_KERNEL,
    ]
    rot = [k for k in preferred if k in kernels]
    if not rot:
        return []

    runs = repository.list_runs()
    if any(
        r.status == RunStatus.QUEUED and r.owner == RESEARCH_AUTOPILOT_OWNER for r in runs
    ):
        return []

    rng = _lab_random()
    kernel = rng.choice(rot)

    names = {r.name for r in runs}
    name = ""
    for _ in range(5000):
        cand = f"autopilot-research-{kernel}-r{rng.randrange(1 << 28)}"
        if cand not in names:
            name = cand
            break
    if not name:
        return []

    if kernel == CPU_BARINA_KERNEL:
        span = rng.randint(55_000, 150_000)
    elif kernel == CPU_ACCELERATED_KERNEL:
        span = rng.randint(22_000, 95_000)
    elif kernel == CPU_PARALLEL_KERNEL:
        span = rng.randint(100_000, 520_000)
    else:
        span = rng.randint(110_000, 380_000)

    sigs = {
        (r.range_start, r.range_end, r.kernel, r.hardware)
        for r in runs
        if r.status
        in {RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.COMPLETED, RunStatus.VALIDATED}
    }

    range_start = 0
    range_end = 0
    for _attempt in range(64):
        base = choose_probe_interval_start(
            kernel, span, rng, loose_upper_base=48_000_000_000
        )
        if base is None:
            return []
        range_start = base
        range_end = base + span - 1
        if range_end > INT64_MAX // 8:
            continue
        if (range_start, range_end, kernel, "cpu") not in sigs:
            break
    else:
        return []

    run = repository.create_run(
        direction_slug="hypothesis-sandbox",
        name=name,
        range_start=range_start,
        range_end=range_end,
        kernel=kernel,
        hardware="cpu",
        owner=RESEARCH_AUTOPILOT_OWNER,
    )
    repository.update_run(
        run.id,
        summary=(
            f"Autopilot research snack ({kernel}) on [{range_start}, {range_end}]. "
            "Exploratory cross-check alongside verification; delete from queue if undesired."
        ),
    )
    return [run.id]


# ---------------------------------------------------------------------------
# Autopilot: randomized probes
# ---------------------------------------------------------------------------


def queue_randomized_compute_runs(
    repository: LabRepository,
    *,
    supported_hardware: list[str] | None = None,
    supported_kernels: list[str] | None = None,
) -> list[str]:
    """Enqueue one small **random** CPU/GPU run when the worker queue would otherwise stay empty.

    Disabled with ``COLLATZ_RANDOM_PROBES=0``. Reproducible with ``COLLATZ_RANDOM_SEED``.
    At most one ``QUEUED`` run with owner :data:`RANDOM_PROBE_OWNER` at a time.
    """
    if not _random_probes_enabled():
        return []
    profile = repository.get_compute_profile()
    if not profile.continuous_enabled:
        return []

    allowed_hw = set(supported_hardware or ["cpu", "gpu"])
    kernels = list(supported_kernels or [])
    runs = repository.list_runs()
    if any(r.status == RunStatus.QUEUED and r.owner == RANDOM_PROBE_OWNER for r in runs):
        return []

    candidates: list[tuple[str, str]] = []
    cpu_pct = _effective_profile_percent(profile, "cpu")
    gpu_pct = _effective_profile_percent(profile, "gpu")
    if "cpu" in allowed_hw and cpu_pct > 0:
        for k in (
            CPU_SIEVE_KERNEL,
            CPU_PARALLEL_ODD_KERNEL,
            CPU_BARINA_KERNEL,
            CPU_PARALLEL_KERNEL,
            CPU_ACCELERATED_KERNEL,
        ):
            if k in kernels:
                candidates.append(("cpu", k))
    if "gpu" in allowed_hw and gpu_pct > 0:
        for k in (GPU_SIEVE_KERNEL, GPU_KERNEL):
            if k in kernels:
                candidates.append(("gpu", k))
    if not candidates:
        return []

    rng = _lab_random()
    hardware, kernel = rng.choice(candidates)

    if kernel == GPU_SIEVE_KERNEL:
        span = rng.randint(350_000, 2_800_000)
    elif kernel == GPU_KERNEL:
        span = rng.randint(12_000, 120_000)
    elif kernel == CPU_BARINA_KERNEL:
        span = rng.randint(45_000, 160_000)
    elif kernel == CPU_ACCELERATED_KERNEL:
        span = rng.randint(18_000, 95_000)
    elif kernel == CPU_PARALLEL_KERNEL:
        span = rng.randint(90_000, 520_000)
    elif kernel == CPU_PARALLEL_ODD_KERNEL:
        span = rng.randint(90_000, 520_000)
    else:
        span = rng.randint(100_000, 650_000)

    sigs = {
        (r.range_start, r.range_end, r.kernel, r.hardware)
        for r in runs
        if r.status
        in {RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.COMPLETED, RunStatus.VALIDATED}
    }
    names = {r.name for r in runs}

    range_start = 0
    range_end = 0
    name = ""
    for _attempt in range(80):
        base = choose_probe_interval_start(
            kernel, span, rng, loose_upper_base=52_000_000_000
        )
        if base is None:
            continue
        range_start = base
        range_end = base + span - 1
        if range_end > INT64_MAX // 8:
            continue
        if (range_start, range_end, kernel, hardware) in sigs:
            continue
        tag = rng.randrange(1 << 28)
        cand = f"random-probe-{kernel}-{tag}"
        if cand in names:
            continue
        name = cand
        break
    if not name:
        return []

    direction = (
        "verification"
        if kernel in {CPU_SIEVE_KERNEL, GPU_SIEVE_KERNEL, CPU_PARALLEL_ODD_KERNEL}
        else "hypothesis-sandbox"
    )
    run = repository.create_run(
        direction_slug=direction,
        name=name,
        range_start=range_start,
        range_end=range_end,
        kernel=kernel,
        hardware=hardware,
        owner=RANDOM_PROBE_OWNER,
    )
    repository.update_run(
        run.id,
        summary=(
            f"Randomized auto probe ({kernel} / {hardware}) on [{range_start}, {range_end}]. "
            "Disable with COLLATZ_RANDOM_PROBES=0."
        ),
    )
    return [run.id]


# ---------------------------------------------------------------------------
# Legacy validation reruns
# ---------------------------------------------------------------------------


def _is_legacy_validation_failure(run: Run) -> bool:
    return (
        run.direction_slug == "verification"
        and run.status == RunStatus.FAILED
        and run.owner != LEGACY_VALIDATION_RERUN_OWNER
        and str(run.summary or "").startswith("Validation failed:")
    )


def queue_legacy_validation_reruns(
    repository: LabRepository,
    *,
    supported_hardware: list[str] | None = None,
    limit: int = 4,
) -> list[str]:
    allowed_hardware = set(supported_hardware or ["cpu", "gpu"])
    runs = repository.list_runs()
    existing_names = {run.name for run in runs}
    existing_exact_ranges = {
        (run.range_start, run.range_end, run.kernel, run.hardware)
        for run in runs
        if run.status in {RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.COMPLETED, RunStatus.VALIDATED}
    }
    queued_ids: list[str] = []

    candidates = sorted(
        [
            run
            for run in runs
            if _is_legacy_validation_failure(run)
            and run.hardware in allowed_hardware
            and f"{LEGACY_VALIDATION_RERUN_PREFIX}{run.id}" not in existing_names
            and (run.range_start, run.range_end, run.kernel, run.hardware) not in existing_exact_ranges
        ],
        key=lambda run: (run.range_start, run.created_at or "", run.id),
    )

    for source in candidates[: max(0, limit)]:
        rerun = repository.create_run(
            direction_slug=source.direction_slug,
            name=f"{LEGACY_VALIDATION_RERUN_PREFIX}{source.id}",
            range_start=source.range_start,
            range_end=source.range_end,
            kernel=source.kernel,
            owner=LEGACY_VALIDATION_RERUN_OWNER,
            code_version=source.code_version,
            hardware=source.hardware,
        )
        queued_ids.append(rerun.id)
        existing_names.add(rerun.name)
        existing_exact_ranges.add((source.range_start, source.range_end, source.kernel, source.hardware))
        if "Legacy revalidation queued via" not in (source.summary or ""):
            repository.update_run(
                source.id,
                summary=(
                    f"{source.summary.rstrip()} "
                    f"Legacy revalidation queued via {rerun.id}."
                ).strip(),
            )

    return queued_ids


def annotate_legacy_validation_failures(
    repository: LabRepository,
    *,
    limit: int = 64,
) -> int:
    runs = repository.list_runs()
    successors_by_signature: dict[tuple[int, int, str, str], list[Run]] = {}
    for run in runs:
        if run.status not in {RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.COMPLETED, RunStatus.VALIDATED}:
            continue
        signature = (run.range_start, run.range_end, run.kernel, run.hardware)
        successors_by_signature.setdefault(signature, []).append(run)

    updated = 0
    for failed in runs:
        if not _is_legacy_validation_failure(failed):
            continue
        if "Superseded by" in (failed.summary or ""):
            continue
        signature = (failed.range_start, failed.range_end, failed.kernel, failed.hardware)
        successors = [
            run
            for run in successors_by_signature.get(signature, [])
            if run.id != failed.id
        ]
        if not successors:
            continue
        successors.sort(
            key=lambda run: (
                0 if run.status == RunStatus.VALIDATED else 1,
                0 if run.status == RunStatus.COMPLETED else 1,
                run.created_at or "",
                run.id,
            )
        )
        successor = successors[0]
        repository.update_run(
            failed.id,
            summary=(
                f"{failed.summary.rstrip()} "
                f"Superseded by {successor.id} ({successor.status.value}) on the current code path."
            ).strip(),
        )
        updated += 1
        if updated >= max(0, limit):
            break

    return updated
