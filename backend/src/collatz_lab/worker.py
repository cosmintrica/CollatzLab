from __future__ import annotations

import json
import time
from dataclasses import dataclass

from .hardware import discover_hardware, select_worker_execution_profile
from .repository import LabRepository
from .schemas import Worker
from .services import process_next_queued_run


@dataclass
class WorkerLoopResult:
    worker: Worker
    processed_run_id: str | None


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
            print(json.dumps({"worker_id": worker.id, "processed_run_id": run.id}))
    except KeyboardInterrupt:
        repository.update_worker(worker.id, status="offline")
        raise
