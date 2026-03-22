from __future__ import annotations

from collatz_lab import hardware, services
from collatz_lab.hardware import discover_hardware, select_worker_execution_profile
from collatz_lab.repository import LabRepository
from collatz_lab.schemas import RunStatus, WorkerStatus
from collatz_lab.services import process_next_queued_run


def test_hardware_discovery_exposes_cpu_capability():
    capabilities = discover_hardware()
    cpu_capabilities = [item for item in capabilities if item.kind == "cpu"]
    assert cpu_capabilities
    assert "cpu-direct" in cpu_capabilities[0].supported_kernels


def test_hardware_discovery_exposes_gpu_kernel_when_runtime_is_ready(monkeypatch):
    monkeypatch.setattr(hardware, "gpu_execution_ready", lambda: True)
    monkeypatch.setattr(
        hardware,
        "_run_command",
        lambda command: "0, RTX 4060 Ti, 16380, 595.71",
    )

    capabilities = discover_hardware()
    gpu_capabilities = [item for item in capabilities if item.kind == "gpu"]
    assert gpu_capabilities
    assert "gpu-collatz-accelerated" in gpu_capabilities[0].supported_kernels


def test_worker_claims_and_executes_queued_run(repository: LabRepository):
    run = repository.create_run(
        direction_slug="verification",
        name="queued-baseline",
        range_start=1,
        range_end=30,
        kernel="cpu-accelerated",
        hardware="cpu",
    )
    capabilities = discover_hardware()
    supported_hardware, supported_kernels = select_worker_execution_profile(
        capabilities,
        requested_hardware="cpu",
    )
    worker = repository.register_worker(
        name="cpu-worker",
        role="compute-agent",
        hardware="cpu",
        capabilities=[item.model_dump() for item in capabilities],
    )

    completed = process_next_queued_run(
        repository,
        worker_id=worker.id,
        supported_hardware=supported_hardware,
        supported_kernels=supported_kernels,
    )

    assert completed is not None
    assert completed.id == run.id
    assert completed.status == RunStatus.COMPLETED
    refreshed_worker = repository.get_worker(worker.id)
    assert refreshed_worker.status == WorkerStatus.IDLE
    assert refreshed_worker.current_run_id is None


def test_gpu_kernel_dispatch_is_used_when_requested(monkeypatch):
    sentinel = services.AggregateMetrics(
        processed=3,
        last_processed=3,
        max_total_stopping_time={"n": 3, "value": 5},
        max_stopping_time={"n": 3, "value": 2},
        max_excursion={"n": 3, "value": 10},
        sample_records=[],
    )

    monkeypatch.setattr(services, "gpu_execution_ready", lambda: True)
    monkeypatch.setattr(services, "compute_range_metrics_gpu", lambda *args, **kwargs: sentinel)

    result = services.compute_range_metrics(1, 3, kernel="gpu-collatz-accelerated")

    assert result is sentinel


def test_register_worker_reuses_existing_name(repository: LabRepository):
    capabilities = discover_hardware()
    first = repository.register_worker(
        name="shared-worker",
        role="compute-agent",
        hardware="cpu",
        capabilities=[item.model_dump() for item in capabilities],
    )
    second = repository.register_worker(
        name="shared-worker",
        role="validator-agent",
        hardware="auto",
        capabilities=[item.model_dump() for item in capabilities],
    )

    assert first.id == second.id
    workers = [worker for worker in repository.list_workers() if worker.name == "shared-worker"]
    assert len(workers) == 1
    assert workers[0].role == "validator-agent"


def test_worker_can_seed_continuous_run_when_queue_is_empty(repository: LabRepository, monkeypatch):
    capabilities = discover_hardware()
    supported_hardware, supported_kernels = select_worker_execution_profile(
        capabilities,
        requested_hardware="cpu",
    )
    worker = repository.register_worker(
        name="continuous-cpu-worker",
        role="compute-agent",
        hardware="cpu",
        capabilities=[item.model_dump() for item in capabilities],
    )

    def fake_execute_run(repo, run_id, checkpoint_interval=250):
        return repo.update_run(run_id, status=RunStatus.COMPLETED, summary="continuous run completed")

    monkeypatch.setattr(services, "execute_run", fake_execute_run)

    completed = process_next_queued_run(
        repository,
        worker_id=worker.id,
        supported_hardware=supported_hardware,
        supported_kernels=supported_kernels,
    )

    assert completed is not None
    assert completed.name == "autopilot-continuous-cpu"
    assert completed.owner == "gemini-autopilot"
    assert completed.hardware == "cpu"


def test_overflow_failure_creates_recovered_prefix_and_exact_cpu_tail(repository: LabRepository):
    failed = repository.create_run(
        direction_slug="verification",
        name="overflow-gpu-run",
        range_start=100,
        range_end=199,
        kernel="gpu-collatz-accelerated",
        hardware="gpu",
    )
    repository.update_run(
        failed.id,
        status=RunStatus.FAILED,
        checkpoint={"next_value": 150, "last_processed": 149},
        metrics={
            "processed": 50,
            "last_processed": 149,
            "max_total_stopping_time": {"n": 123, "value": 44},
            "max_stopping_time": {"n": 121, "value": 22},
            "max_excursion": {"n": 141, "value": 999},
            "sample_records": [],
        },
        summary="Worker COL-0001 failed: GPU int64 overflow guard triggered at seed 155.",
    )

    created_ids = services._ensure_overflow_recovery_runs(repository, run_ids={failed.id})

    assert len(created_ids) == 2
    prefix = next(run for run in repository.list_runs() if run.name == f"recover-prefix-{failed.id}")
    tail = next(run for run in repository.list_runs() if run.name == f"recover-tail-{failed.id}")
    assert prefix.status == RunStatus.COMPLETED
    assert prefix.range_start == 100
    assert prefix.range_end == 149
    assert prefix.hardware == "gpu"
    assert tail.status == RunStatus.QUEUED
    assert tail.range_start == 150
    assert tail.range_end == 199
    assert tail.hardware == "cpu"
    assert tail.kernel == "cpu-parallel"


def test_continuous_queue_uses_parallel_kernel_after_cpu_overflow(repository: LabRepository):
    completed = repository.create_run(
        direction_slug="verification",
        name="completed-baseline",
        range_start=1,
        range_end=100,
        kernel="cpu-parallel",
        hardware="cpu",
    )
    repository.update_run(
        completed.id,
        status=RunStatus.COMPLETED,
        checkpoint={"next_value": 101, "last_processed": 100},
        metrics={"processed": 100, "last_processed": 100},
        summary="Completed baseline interval.",
    )
    failed = repository.create_run(
        direction_slug="verification",
        name="overflow-cpu-run",
        range_start=101,
        range_end=200,
        kernel="cpu-parallel",
        hardware="cpu",
    )
    repository.update_run(
        failed.id,
        status=RunStatus.FAILED,
        checkpoint={"next_value": 151, "last_processed": 150},
        metrics={
            "processed": 50,
            "last_processed": 150,
            "max_total_stopping_time": {"n": 140, "value": 17},
            "max_stopping_time": {"n": 133, "value": 9},
            "max_excursion": {"n": 149, "value": 1000},
            "sample_records": [],
        },
        summary="Worker COL-0002 failed: CPU parallel int64 overflow guard triggered at seed 160.",
    )

    services._ensure_overflow_recovery_runs(repository, run_ids={failed.id})
    tail = next(run for run in repository.list_runs() if run.name == f"recover-tail-{failed.id}")
    repository.update_run(
        tail.id,
        status=RunStatus.COMPLETED,
        checkpoint={"next_value": 201, "last_processed": 200},
        metrics={"processed": 50, "last_processed": 200},
        summary="Recovered exact tail.",
    )

    queued_ids = services.queue_continuous_verification_runs(repository, supported_hardware=["cpu"])

    assert queued_ids
    next_run = repository.get_run(queued_ids[0])
    assert next_run.range_start == 201
    # Continuous CPU runs now use odd-only kernel for 2x throughput.
    assert next_run.kernel == "cpu-parallel-odd"
    assert next_run.name == "autopilot-continuous-cpu"


def test_duplicate_queued_recovery_tail_is_pruned(repository: LabRepository):
    failed = repository.create_run(
        direction_slug="verification",
        name="overflow-prune-run",
        range_start=100,
        range_end=199,
        kernel="cpu-parallel",
        hardware="cpu",
    )
    repository.update_run(
        failed.id,
        status=RunStatus.FAILED,
        checkpoint={"next_value": 150, "last_processed": 149},
        metrics={
            "processed": 50,
            "last_processed": 149,
            "max_total_stopping_time": {"n": 140, "value": 17},
            "max_stopping_time": {"n": 133, "value": 9},
            "max_excursion": {"n": 149, "value": 1000},
            "sample_records": [],
        },
        summary="Worker COL-0003 failed: CPU parallel int64 overflow guard triggered at seed 160.",
    )

    services._ensure_overflow_recovery_runs(repository, run_ids={failed.id})
    duplicate = repository.create_run(
        direction_slug="verification",
        name=f"recover-tail-{failed.id}",
        range_start=150,
        range_end=199,
        kernel="cpu-parallel",
        hardware="cpu",
        owner="overflow-recovery",
    )
    assert duplicate.status == RunStatus.QUEUED

    services._ensure_overflow_recovery_runs(repository, run_ids={failed.id})

    tails = [run for run in repository.list_runs() if run.name == f"recover-tail-{failed.id}"]
    assert len(tails) == 1


def test_cpu_accelerated_uses_larger_checkpoint_batches(repository: LabRepository):
    assert services._effective_checkpoint_interval("cpu-accelerated", 250) >= 50_000


def test_api_can_enqueue_run_without_immediate_execution(settings):
    from fastapi.testclient import TestClient

    from collatz_lab.api import create_app

    app = create_app(settings)
    client = TestClient(app)

    response = client.post(
        "/api/runs",
        json={
            "direction_slug": "verification",
            "name": "queued-from-api",
            "range_start": 10,
            "range_end": 40,
            "enqueue_only": True,
            "kernel": "cpu-direct",
            "hardware": "cpu",
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "queued"

    workers_response = client.get("/api/workers/capabilities")
    assert workers_response.status_code == 200
    assert any(item["kind"] == "cpu" for item in workers_response.json())
