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
