from __future__ import annotations

import json

from collatz_lab.autopilot import run_autopilot_cycle
from collatz_lab.repository import LabRepository
from collatz_lab.schemas import LLMAutopilotRun, RunStatus, TaskStatus
from collatz_lab.services import execute_run
from collatz_lab.task_execution import execute_supported_tasks, execute_task


def test_execute_inverse_tree_task_creates_real_artifacts(repository: LabRepository):
    run = repository.create_run(
        direction_slug="verification",
        name="seed-records",
        range_start=1,
        range_end=60,
        kernel="cpu-accelerated",
        hardware="cpu",
    )
    completed = execute_run(repository, run.id, checkpoint_interval=15)
    assert completed.status == RunStatus.COMPLETED

    task = repository.create_task(
        direction_slug="inverse-tree-parity",
        title="Analyze parity vectors of all consolidated record-breaking numbers",
        kind="analysis",
        description="Turn the current run records into parity and residue summaries.",
        owner="theory-agent",
        priority=1,
    )

    result = execute_task(repository, task.id)

    assert result.executed is True
    assert result.artifact_ids
    assert repository.get_task(task.id).status == TaskStatus.DONE
    created_artifacts = [artifact for artifact in repository.list_artifacts() if artifact.id in result.artifact_ids]
    assert any(artifact.metadata.get("task_id") == task.id for artifact in created_artifacts)
    json_artifact = next(artifact for artifact in created_artifacts if artifact.kind.value == "json")
    artifact_path = repository.settings.workspace_root / json_artifact.path
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert "record_entries" not in payload
    assert "unique_seeds" not in payload
    assert payload["record_entry_count"] >= len(payload["record_entry_preview"])


def test_autopilot_apply_executes_supported_tasks(repository: LabRepository, monkeypatch):
    run = repository.create_run(
        direction_slug="verification",
        name="autopilot-records",
        range_start=1,
        range_end=45,
        kernel="cpu-accelerated",
        hardware="cpu",
    )
    execute_run(repository, run.id, checkpoint_interval=10)

    def fake_plan(**_kwargs):
        return LLMAutopilotRun.model_validate(
            {
                "provider": "gemini",
                "model": "gemini-3-flash-preview",
                "safety_mode": "review_only",
                "created_at": "2026-03-22T00:00:00+00:00",
                "summary": "Focus on structure and claim consolidation.",
                "recommended_direction_slug": "lemma-workspace",
                "recommended_direction_reason": "Use existing evidence before more compute.",
                "memory_mode": "local_history",
                "notes": ["Work from the current database only."],
                "task_proposals": [
                    {
                        "direction_slug": "lemma-workspace",
                        "title": "Consolidate all max_total_stopping_time records into claim COL-0020",
                        "kind": "analysis",
                        "description": "Use completed runs to update the current record-focused claim.",
                        "priority": 1,
                    }
                ],
            }
        )

    monkeypatch.setattr("collatz_lab.autopilot.generate_autopilot_plan", fake_plan)

    result = run_autopilot_cycle(repository, max_tasks=1, apply=True, mode="manual")

    assert result.applied is True
    assert result.applied_task_ids
    assert result.executed_task_ids
    assert result.generated_artifact_ids
    applied_task = repository.get_task(result.applied_task_ids[0])
    assert applied_task.status == TaskStatus.DONE


def test_execute_two_adic_task_creates_valuation_artifacts(repository: LabRepository):
    run = repository.create_run(
        direction_slug="verification",
        name="seed-two-adic-records",
        range_start=1,
        range_end=75,
        kernel="cpu-accelerated",
        hardware="cpu",
    )
    execute_run(repository, run.id, checkpoint_interval=20)

    task = repository.create_task(
        direction_slug="two-adic-v2",
        title="Profile v2(3n+1) on current record seeds",
        kind="analysis",
        description="Summarize odd-step valuations on record seeds.",
        owner="theory-agent",
        priority=1,
    )

    result = execute_task(repository, task.id)

    assert result.executed is True
    assert result.artifact_ids
    created_artifacts = [artifact for artifact in repository.list_artifacts() if artifact.id in result.artifact_ids]
    json_artifact = next(artifact for artifact in created_artifacts if artifact.kind.value == "json")
    artifact_path = repository.settings.workspace_root / json_artifact.path
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["analysis_type"] == "two-adic-v2"
    assert payload["v2_histogram"]
    assert payload["odd_seed_mod_16"]


def test_execute_supported_tasks_skips_unsupported_when_no_data(repository: LabRepository):
    task = repository.create_task(
        direction_slug="verification",
        title="Characterize numbers with extreme excursion values",
        kind="analysis",
        description="Should wait for completed runs.",
    )

    batch = execute_supported_tasks(repository, limit=2, preferred_task_ids=[task.id])

    assert batch.executed_task_ids == []
    assert repository.get_task(task.id).status == TaskStatus.OPEN
