from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from .hardware import discover_hardware, select_worker_execution_profile
from .hardware.constants import (
    CPU_ACCELERATED_KERNEL,
    CPU_BARINA_KERNEL,
    CPU_DIRECT_KERNEL,
    CPU_PARALLEL_ODD_KERNEL,
    CPU_SIEVE_KERNEL,
    GPU_KERNEL,
    GPU_SIEVE_KERNEL,
)
from .metrics_sot import INT64_MAX
from .repository import LabRepository
from .schemas import RunStatus, Worker, WorkerStatus
from .scheduling import _lab_random, choose_probe_interval_start
from .services import process_next_queued_run
from .validation import validate_run


# ---------------------------------------------------------------------------
# Intelligent work scheduling
# ---------------------------------------------------------------------------
# Between verification runs, the worker opportunistically:
#   1. Auto-validates recently completed runs (selective validation)
#   2. Runs hypothesis experiments (residue class, record seeds, etc.)
# This ensures the system does more than brute-force verification.

VALIDATION_INTERVAL = 5       # validate after every N verification runs
MAX_CONSECUTIVE_FAILURES = 5  # stop retrying after N consecutive failures
FAILURE_BACKOFF_BASE = 10     # seconds to wait after a failure (doubles each time)

# Small queued runs under hypothesis-sandbox to rotate kernels (incl. cpu-barina) without manual ops.
KERNEL_PROBE_OWNER = "kernel-probe-worker"

_KERNEL_PROBE_ORDER: tuple[str, ...] = (
    CPU_SIEVE_KERNEL,
    CPU_BARINA_KERNEL,
    CPU_PARALLEL_ODD_KERNEL,
    CPU_ACCELERATED_KERNEL,
    CPU_DIRECT_KERNEL,
    GPU_SIEVE_KERNEL,
    GPU_KERNEL,
)


def _hypothesis_every_n_runs() -> int:
    """Completed verification runs between sandbox experiments (CPU worker)."""
    try:
        return max(1, int(os.getenv("COLLATZ_HYPOTHESIS_EVERY_N_RUNS", "8").strip()))
    except ValueError:
        return 8


def _hypothesis_every_n_runs_gpu() -> int:
    """Same for GPU-profile workers (analysis is still CPU-side Python)."""
    try:
        return max(1, int(os.getenv("COLLATZ_HYPOTHESIS_EVERY_N_RUNS_GPU", "14").strip()))
    except ValueError:
        return 14


def _hypothesis_idle_polls() -> int:
    """Consecutive idle polls with no queued run before one sandbox experiment. 0 = disabled."""
    try:
        v = int(os.getenv("COLLATZ_HYPOTHESIS_IDLE_POLLS", "30").strip())
        return max(0, v)
    except ValueError:
        return 30


def _hypothesis_rotation_sequence() -> list[int]:
    """Sandbox experiment slot indices (0..6). Env ``COLLATZ_HYPOTHESIS_ROTATION_MODE``:

    - ``full`` (default): residue, records, trajectory, growth, mod3, stratified, glide
    - ``core``: first five only (no stratified / glide auto-runs)
    - ``minimal``: residue, growth, mod3 only
    """
    mode = os.getenv("COLLATZ_HYPOTHESIS_ROTATION_MODE", "full").strip().lower()
    if mode == "core":
        return [0, 1, 2, 3, 4]
    if mode == "minimal":
        return [0, 3, 4]
    return [0, 1, 2, 3, 4, 5, 6]


def _hypothesis_rotation_mode_label() -> str:
    return os.getenv("COLLATZ_HYPOTHESIS_ROTATION_MODE", "full").strip().lower() or "full"


def _kernel_probe_every_n_runs() -> int:
    """Completed verification runs between auto-queued kernel probe runs. 0 = disabled."""
    try:
        v = int(os.getenv("COLLATZ_KERNEL_PROBE_EVERY_N_RUNS", "28").strip())
        return max(0, v)
    except ValueError:
        return 28


def _hypothesis_cursor_path(repository: LabRepository, worker_name: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in worker_name.strip())
    return repository.settings.data_dir / f"hypothesis_sandbox_cursor_{safe or 'default'}.json"


def _load_sandbox_cursor(repository: LabRepository, worker_name: str) -> tuple[int, int]:
    """Load hypothesis rotation counter + kernel-probe rotation index (persistent)."""
    path = _hypothesis_cursor_path(repository, worker_name)
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            hr = max(0, int(data.get("hypotheses_run", 0)))
            kp = max(0, int(data.get("kernel_probe_index", 0)))
            return hr, kp
    except Exception:
        pass
    return 0, 0


@dataclass
class WorkerLoopResult:
    worker: Worker
    processed_run_id: str | None


@dataclass
class _WorkerState:
    """Tracks worker activity for intelligent scheduling."""
    runs_since_validation: int = 0
    runs_since_hypothesis: int = 0
    runs_since_kernel_probe: int = 0
    total_runs: int = 0
    validations_done: int = 0
    hypotheses_run: int = 0
    kernel_probe_index: int = 0
    consecutive_failures: int = 0
    idle_poll_streak: int = 0


def _save_sandbox_cursor(repository: LabRepository, worker_name: str, state: _WorkerState) -> None:
    path = _hypothesis_cursor_path(repository, worker_name)
    repository.settings.data_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "hypotheses_run": state.hypotheses_run,
                "kernel_probe_index": state.kernel_probe_index,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _try_auto_validate(repository: LabRepository, state: _WorkerState) -> bool:
    """Validate the highest-priority unvalidated completed run.

    Priority: smallest range first (fastest to validate), then oldest first.
    This ensures small runs don't remain unvalidated while large recent runs
    monopolize the validation queue — the systematic-bug gap is worst for
    runs with no new records, which are never cross-checked by bigint SoT.
    """
    try:
        runs = repository.list_runs()
        candidates = [
            r for r in runs
            if r.status == RunStatus.COMPLETED
            and r.direction_slug == "verification"
        ]
        # Smallest range first (validates fastest), oldest first as tiebreaker.
        candidates.sort(key=lambda r: (r.range_end - r.range_start, r.created_at))
        if not candidates:
            return False

        target = candidates[0]
        print(json.dumps({
            "action": "auto-validate",
            "run_id": target.id,
            "range": f"{target.range_start}-{target.range_end}",
            "kernel": target.kernel,
        }))
        validated = validate_run(repository, target.id)
        state.validations_done += 1
        print(json.dumps({
            "action": "auto-validate-done",
            "run_id": target.id,
            "status": validated.status.value,
            "summary": validated.summary[:200],
        }))
        return True
    except Exception as exc:
        print(json.dumps({"action": "auto-validate-error", "error": str(exc)}))
        return False


def _try_hypothesis_experiment(
    repository: LabRepository, state: _WorkerState, *, worker_name: str
) -> bool:
    """Run a quick hypothesis experiment between verification work."""
    try:
        from .hypothesis import (
            analyze_glide_structure,
            analyze_residue_classes,
            analyze_record_seeds,
            analyze_residue_classes_stratified,
            scan_trajectory_depths,
            test_mod3_convergence_redundancy,
            test_stopping_time_growth,
            DIRECTION_SLUG,
        )
        from .schemas import ArtifactKind

        rotation_seq = _hypothesis_rotation_sequence()
        experiment_idx = rotation_seq[state.hypotheses_run % len(rotation_seq)]
        rotation_mode = _hypothesis_rotation_mode_label()
        end = min(50_000, 10_000 * (state.hypotheses_run + 1))

        if experiment_idx == 0:
            modulus = [3, 4, 6, 8, 12, 16, 24, 32][state.hypotheses_run % 8]
            result = analyze_residue_classes(modulus, start=1, end=end)
        elif experiment_idx == 1:
            result = analyze_record_seeds(start=1, end=end)
        elif experiment_idx == 2:
            result = scan_trajectory_depths(start=1, end=end)
        elif experiment_idx == 3:
            result = test_stopping_time_growth(start=1, end=end)
        elif experiment_idx == 4:
            result = test_mod3_convergence_redundancy(start=1, end=min(end, 10_000))
        elif experiment_idx == 5:
            result = analyze_residue_classes_stratified(8, start=1, end=end, bin_count=8)
        else:
            result = analyze_glide_structure(1, end, sample_cap=4_000)

        # Store as claim under hypothesis-sandbox
        claim = repository.create_claim(
            direction_slug=DIRECTION_SLUG,
            title=result.title,
            statement=result.statement,
            owner="hypothesis-sandbox",
            notes=result.notes,
        )
        status_map = {
            "proposed": "idea",
            "testing": "active",
            "plausible": "promising",
            "falsified": "refuted",
        }
        mapped = status_map.get(result.status, "idea")
        repository.update_claim_status(claim.id, mapped)

        # Save evidence artifact
        evidence_path = (
            repository.settings.artifacts_dir
            / "hypotheses"
            / f"{claim.id}-evidence.json"
        )
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        import json as _json
        payload = {
            "hypothesis_id": claim.id,
            "category": result.category,
            "status": result.status,
            "test_methodology": result.test_methodology,
            "test_range": result.test_range,
            "evidence": result.evidence,
            "falsification": getattr(result, "falsification", "") or "",
            "origin": "hypothesis-worker-rotation",
            "rotation_mode": rotation_mode,
        }
        evidence_path.write_text(_json.dumps(payload, indent=2, default=str), encoding="utf-8")
        repository.create_artifact(
            kind=ArtifactKind.JSON,
            path=evidence_path,
            claim_id=claim.id,
            metadata={
                "type": "hypothesis-evidence",
                "category": result.category,
                "origin": "hypothesis-worker-rotation",
                "rotation_mode": rotation_mode,
            },
        )

        state.hypotheses_run += 1
        _save_sandbox_cursor(repository, worker_name, state)
        print(json.dumps({
            "action": "hypothesis-experiment",
            "claim_id": claim.id,
            "category": result.category,
            "status": result.status,
            "title": result.title[:100],
        }))
        return True
    except Exception as exc:
        print(json.dumps({"action": "hypothesis-error", "error": str(exc), "tb": traceback.format_exc()[:500]}))
        return False


def _kernel_run_hardware(kernel: str) -> str:
    return "gpu" if kernel in (GPU_SIEVE_KERNEL, GPU_KERNEL) else "cpu"


def _ordered_probe_kernels(hardware: str, supported_kernels: list[str]) -> list[str]:
    """Kernels this worker may enqueue for periodic probes, stable breadth-first order."""
    supported = set(supported_kernels)
    out: list[str] = []
    for k in _KERNEL_PROBE_ORDER:
        if k not in supported:
            continue
        if _kernel_run_hardware(k) != hardware:
            continue
        out.append(k)
    return out


def _try_enqueue_kernel_probe(
    repository: LabRepository,
    state: _WorkerState,
    *,
    worker_name: str,
    hardware: str,
    supported_kernels: list[str],
) -> bool:
    """Queue one small hypothesis-sandbox run to rotate kernels (incl. Barina). Best-effort."""
    if _kernel_probe_every_n_runs() <= 0:
        return False
    kernels = _ordered_probe_kernels(hardware, supported_kernels)
    if not kernels:
        return False

    runs = repository.list_runs()
    if any(
        r.status == RunStatus.QUEUED and r.owner == KERNEL_PROBE_OWNER for r in runs
    ):
        return False

    sigs = {
        (r.range_start, r.range_end, r.kernel, r.hardware)
        for r in runs
        if r.status
        in {RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.COMPLETED, RunStatus.VALIDATED}
    }

    idx = state.kernel_probe_index % len(kernels)
    kernel = kernels[idx]
    run_hw = _kernel_run_hardware(kernel)
    rng = _lab_random()

    if kernel == CPU_DIRECT_KERNEL:
        span = rng.randint(400, 1_200)
    elif kernel == CPU_BARINA_KERNEL:
        span = rng.randint(35_000, 95_000)
    elif kernel == CPU_ACCELERATED_KERNEL:
        span = rng.randint(18_000, 72_000)
    elif kernel in (GPU_SIEVE_KERNEL, GPU_KERNEL):
        span = rng.randint(120_000, 420_000)
    else:
        span = rng.randint(55_000, 220_000)

    range_start = 0
    range_end = 0
    for _ in range(48):
        base = choose_probe_interval_start(
            kernel, span, rng, loose_upper_base=40_000_000_000
        )
        if base is None:
            return False
        range_start = base
        range_end = base + span - 1
        if range_end > INT64_MAX // 8:
            continue
        if (range_start, range_end, kernel, run_hw) not in sigs:
            break
    else:
        return False

    names = {r.name for r in runs}
    name = ""
    for _ in range(5000):
        cand = f"kernel-probe-{kernel}-r{rng.randrange(1 << 28)}"
        if cand not in names:
            name = cand
            break
    if not name:
        return False

    run = repository.create_run(
        direction_slug="hypothesis-sandbox",
        name=name,
        range_start=range_start,
        range_end=range_end,
        kernel=kernel,
        hardware=run_hw,
        owner=KERNEL_PROBE_OWNER,
    )
    repository.update_run(
        run.id,
        summary=(
            f"Periodic kernel probe ({kernel}) on [{range_start}, {range_end}]. "
            "Tune via COLLATZ_KERNEL_PROBE_EVERY_N_RUNS (0 disables)."
        ),
    )
    state.kernel_probe_index = idx + 1
    _save_sandbox_cursor(repository, worker_name, state)
    print(
        json.dumps(
            {
                "action": "kernel-probe-queued",
                "run_id": run.id,
                "kernel": kernel,
                "hardware": run_hw,
                "range": f"{range_start}-{range_end}",
            }
        )
    )
    return True


def start_worker_loop(
    repository: LabRepository,
    *,
    name: str,
    role: str,
    hardware: str,
    poll_interval: float,
    validate_after_run: bool,
    once: bool,
) -> WorkerLoopResult:
    capabilities = discover_hardware()
    supported_hardware, supported_kernels = select_worker_execution_profile(
        capabilities,
        requested_hardware=hardware,
    )
    worker = repository.register_worker(
        name=name,
        role=role,
        hardware=hardware,
        capabilities=[cap.model_dump() for cap in capabilities],
    )
    hr, kp = _load_sandbox_cursor(repository, name)
    state = _WorkerState(hypotheses_run=hr, kernel_probe_index=kp)

    try:
        while True:
            run = process_next_queued_run(
                repository,
                worker_id=worker.id,
                supported_hardware=supported_hardware,
                supported_kernels=supported_kernels,
                validate_after_run=validate_after_run,
            )
            if once:
                repository.update_worker(worker.id, status="offline")
                return WorkerLoopResult(worker=repository.get_worker(worker.id), processed_run_id=run.id if run else None)
            if run is None:
                repository.update_worker(worker.id, status="idle")
                state.consecutive_failures = 0
                hyp_idle = _hypothesis_idle_polls()
                if hyp_idle > 0:
                    state.idle_poll_streak += 1
                    if state.idle_poll_streak >= hyp_idle:
                        state.idle_poll_streak = 0
                        _try_hypothesis_experiment(repository, state, worker_name=name)
                time.sleep(poll_interval)
                continue

            # Check if run failed (non-overflow failures trigger backoff)
            if run.status == RunStatus.FAILED and "overflow guard" not in (run.summary or "").lower():
                state.consecutive_failures += 1
                backoff = min(FAILURE_BACKOFF_BASE * (2 ** (state.consecutive_failures - 1)), 300)
                print(json.dumps({
                    "worker_id": worker.id,
                    "action": "failure-backoff",
                    "failed_run_id": run.id,
                    "consecutive_failures": state.consecutive_failures,
                    "backoff_seconds": backoff,
                    "summary": (run.summary or "")[:200],
                }))
                if state.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(json.dumps({
                        "worker_id": worker.id,
                        "action": "max-failures-reached",
                        "consecutive_failures": state.consecutive_failures,
                        "message": f"Stopping after {MAX_CONSECUTIVE_FAILURES} consecutive failures. Manual intervention required.",
                    }))
                    repository.update_worker(worker.id, status=WorkerStatus.ERROR)
                    return WorkerLoopResult(worker=repository.get_worker(worker.id), processed_run_id=run.id)
                time.sleep(backoff)
                continue

            state.consecutive_failures = 0
            state.idle_poll_streak = 0
            state.total_runs += 1
            state.runs_since_validation += 1
            state.runs_since_hypothesis += 1
            state.runs_since_kernel_probe += 1
            print(json.dumps({
                "worker_id": worker.id,
                "processed_run_id": run.id,
                "total_runs": state.total_runs,
            }))

            # ── Intelligent work between verification batches ──

            # Auto-validate after every N runs (CPU worker only — GPU
            # should stay on heavy compute)
            if (
                hardware == "cpu"
                and state.runs_since_validation >= VALIDATION_INTERVAL
            ):
                state.runs_since_validation = 0
                _try_auto_validate(repository, state)

            # Hypothesis sandbox: tunable via env; GPU-named workers use the GPU interval.
            hyp_every = (
                _hypothesis_every_n_runs() if hardware == "cpu" else _hypothesis_every_n_runs_gpu()
            )
            if state.runs_since_hypothesis >= hyp_every:
                state.runs_since_hypothesis = 0
                _try_hypothesis_experiment(repository, state, worker_name=name)

            k_every = _kernel_probe_every_n_runs()
            if k_every > 0 and state.runs_since_kernel_probe >= k_every:
                state.runs_since_kernel_probe = 0
                _try_enqueue_kernel_probe(
                    repository,
                    state,
                    worker_name=name,
                    hardware=hardware,
                    supported_kernels=list(supported_kernels),
                )

    except KeyboardInterrupt:
        repository.update_worker(worker.id, status="offline")
        raise
