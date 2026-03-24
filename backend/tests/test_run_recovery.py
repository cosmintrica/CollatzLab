from __future__ import annotations

from collatz_lab.database import connect
from collatz_lab.hardware.constants import CPU_SIEVE_KERNEL, GPU_SIEVE_KERNEL
from collatz_lab.repository import LabRepository
from collatz_lab.schemas import RunStatus, WorkerStatus


def test_release_running_run_to_queue_clears_worker_and_preserves_checkpoint(
    repository: LabRepository,
    settings,
):
    run = repository.create_run(
        direction_slug="verification",
        name="stuck-sim",
        range_start=1,
        range_end=1000,
        kernel="cpu-accelerated",
        hardware="cpu",
    )
    worker = repository.register_worker(
        name="w1",
        role="compute-agent",
        hardware="cpu",
        capabilities=[],
    )
    cp = {"next_value": 42, "last_processed": 41}
    metrics = {"processed": 40, "max_total_stopping_time": {"n": 1, "value": 0}}
    with connect(str(settings.db_path)) as conn:
        conn.execute(
            """
            UPDATE runs SET status = ?, checkpoint_json = ?, metrics_json = ?, started_at = ?
            WHERE id = ?
            """,
            (
                RunStatus.RUNNING.value,
                __import__("json").dumps(cp),
                __import__("json").dumps(metrics),
                "2020-01-01T00:00:00Z",
                run.id,
            ),
        )
        conn.execute(
            """
            UPDATE workers SET status = ?, current_run_id = ? WHERE id = ?
            """,
            (WorkerStatus.RUNNING.value, run.id, worker.id),
        )
        conn.commit()

    out = repository.release_running_run_to_queue(run.id, note="Test release.")
    assert out.status == RunStatus.QUEUED
    assert out.checkpoint.get("next_value") == 42
    assert out.metrics.get("processed") == 40
    assert "Test release." in (out.summary or "")
    w2 = repository.get_worker(worker.id)
    assert w2.current_run_id is None
    assert w2.status == WorkerStatus.IDLE


def test_set_run_kernel_hardware_only_when_queued(repository: LabRepository):
    run = repository.create_run(
        direction_slug="verification",
        name="migrate-me",
        range_start=1,
        range_end=10,
        kernel=GPU_SIEVE_KERNEL,
        hardware="gpu",
    )
    with connect(str(repository.settings.db_path)) as conn:
        conn.execute(
            "UPDATE runs SET status = ? WHERE id = ?",
            (RunStatus.RUNNING.value, run.id),
        )
        conn.commit()
    try:
        repository.set_run_kernel_hardware(run.id, kernel=CPU_SIEVE_KERNEL, hardware="cpu")
    except ValueError as exc:
        assert "queued" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")

    with connect(str(repository.settings.db_path)) as conn:
        conn.execute(
            "UPDATE runs SET status = ? WHERE id = ?",
            (RunStatus.QUEUED.value, run.id),
        )
        conn.commit()

    run2 = repository.set_run_kernel_hardware(
        run.id,
        kernel=CPU_SIEVE_KERNEL,
        hardware="cpu",
    )
    assert run2.kernel == CPU_SIEVE_KERNEL
    assert run2.hardware == "cpu"
