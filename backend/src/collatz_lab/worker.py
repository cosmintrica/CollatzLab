from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field

from .hardware import discover_hardware, select_worker_execution_profile
from .repository import LabRepository
from .schemas import RunStatus, Worker
from .services import process_next_queued_run, validate_run


# ---------------------------------------------------------------------------
# Intelligent work scheduling
# ---------------------------------------------------------------------------
# Between verification runs, the worker opportunistically:
#   1. Auto-validates recently completed runs (selective validation)
#   2. Runs hypothesis experiments (residue class, record seeds, etc.)
# This ensures the system does more than brute-force verification.

VALIDATION_INTERVAL = 5       # validate after every N verification runs
HYPOTHESIS_INTERVAL = 20      # run hypothesis experiment after every N runs


@dataclass
class WorkerLoopResult:
    worker: Worker
    processed_run_id: str | None


@dataclass
class _WorkerState:
    """Tracks worker activity for intelligent scheduling."""
    runs_since_validation: int = 0
    runs_since_hypothesis: int = 0
    total_runs: int = 0
    validations_done: int = 0
    hypotheses_run: int = 0


def _try_auto_validate(repository: LabRepository, state: _WorkerState) -> bool:
    """Validate the most recent unvalidated completed run."""
    try:
        runs = repository.list_runs()
        candidates = [
            r for r in runs
            if r.status == RunStatus.COMPLETED
            and r.direction_slug == "verification"
        ]
        # Pick the most recent unvalidated run
        candidates.sort(key=lambda r: r.created_at, reverse=True)
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


def _try_hypothesis_experiment(repository: LabRepository, state: _WorkerState) -> bool:
    """Run a quick hypothesis experiment between verification work."""
    try:
        from .hypothesis import (
            analyze_residue_classes,
            analyze_record_seeds,
            scan_trajectory_depths,
            test_stopping_time_growth,
            DIRECTION_SLUG,
        )
        from .schemas import ArtifactKind

        # Rotate through different experiment types
        experiment_idx = state.hypotheses_run % 4
        end = min(50_000, 10_000 * (state.hypotheses_run + 1))

        if experiment_idx == 0:
            modulus = [3, 4, 6, 8, 12, 16, 24, 32][state.hypotheses_run % 8]
            result = analyze_residue_classes(modulus, start=1, end=end)
        elif experiment_idx == 1:
            result = analyze_record_seeds(start=1, end=end)
        elif experiment_idx == 2:
            result = scan_trajectory_depths(start=1, end=end)
        else:
            result = test_stopping_time_growth(start=1, end=end)

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
        repository.update_claim_status(claim.id, status_map.get(result.status, "idea"))

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
        }
        evidence_path.write_text(_json.dumps(payload, indent=2, default=str), encoding="utf-8")
        repository.create_artifact(
            kind=ArtifactKind.JSON,
            path=evidence_path,
            claim_id=claim.id,
            metadata={"type": "hypothesis-evidence", "category": result.category},
        )

        state.hypotheses_run += 1
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
    state = _WorkerState()

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
                time.sleep(poll_interval)
                continue

            state.total_runs += 1
            state.runs_since_validation += 1
            state.runs_since_hypothesis += 1
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

            # Run hypothesis experiment periodically (CPU only)
            if (
                hardware == "cpu"
                and state.runs_since_hypothesis >= HYPOTHESIS_INTERVAL
            ):
                state.runs_since_hypothesis = 0
                _try_hypothesis_experiment(repository, state)

    except KeyboardInterrupt:
        repository.update_worker(worker.id, status="offline")
        raise
