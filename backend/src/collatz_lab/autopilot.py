from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

from .llm import generate_autopilot_plan
from .repository import LabRepository, utc_now
from .schemas import ArtifactKind, LLMAutopilotRun, LLMAutopilotStatus
from .scheduling import queue_continuous_verification_runs
from .task_execution import execute_supported_tasks


def _timestamp_slug(value: str) -> str:
    return value.replace(":", "-").replace("+", "_")


def _collect_autopilot_context(repository: LabRepository) -> dict[str, object]:
    return {
        "baseline": repository.consensus_baseline(),
        "directions": [direction.model_dump(mode="json") for direction in repository.list_directions()],
        "tasks": [
            task.model_dump(mode="json")
            for task in repository.list_tasks()
            if task.status in {"open", "in_progress"}
        ],
        "claims": [claim.model_dump(mode="json") for claim in repository.list_claims()],
        "sources": [source.model_dump(mode="json") for source in repository.list_sources()],
        "runs": [run.model_dump(mode="json") for run in repository.list_runs()[:24]],
    }


def _queue_maintenance_runs_if_needed(repository: LabRepository) -> list[str]:
    workers = repository.list_workers()
    idle_hardware = [
        worker.hardware
        for worker in workers
        if worker.status == "idle" and worker.hardware in {"cpu", "gpu"}
    ]
    if not idle_hardware:
        return []
    return queue_continuous_verification_runs(
        repository,
        supported_hardware=idle_hardware,
        owner="gemini-autopilot",
    )


def _apply_task_proposals(repository: LabRepository, plan: LLMAutopilotRun) -> list[str]:
    applied_task_ids: list[str] = []
    existing_tasks = [
        task
        for task in repository.list_tasks()
        if task.status in {"open", "in_progress"}
    ]
    existing_signatures = {
        (task.direction_slug.lower(), task.title.strip().lower())
        for task in existing_tasks
    }
    for proposal in plan.task_proposals:
        signature = (proposal.direction_slug.lower(), proposal.title.strip().lower())
        if signature in existing_signatures:
            continue
        created = repository.create_task(
            direction_slug=proposal.direction_slug,
            title=proposal.title,
            kind=proposal.kind,
            description=proposal.description,
            owner="gemini-autopilot",
            priority=proposal.priority,
        )
        applied_task_ids.append(created.id)
        existing_signatures.add(signature)
    return applied_task_ids


def _write_autopilot_report(
    repository: LabRepository,
    *,
    plan: LLMAutopilotRun,
    applied_task_ids: list[str],
) -> tuple[str | None, str | None]:
    report_path = repository.settings.reports_dir / f"gemini-autopilot-{_timestamp_slug(plan.created_at)}.md"
    task_lines = []
    for index, proposal in enumerate(plan.task_proposals, start=1):
        task_lines.append(
            f"{index}. [{proposal.direction_slug}] {proposal.title}\n"
            f"   - Kind: {proposal.kind}\n"
            f"   - Priority: {proposal.priority}\n"
            f"   - Description: {proposal.description}"
        )
    notes = "\n".join(f"- {item}" for item in plan.notes) or "- none"
    tasks_block = "\n".join(task_lines) if task_lines else "No task proposals."
    applied_block = ", ".join(applied_task_ids) if applied_task_ids else "none"
    executed_block = ", ".join(plan.executed_task_ids) if plan.executed_task_ids else "none"
    artifact_block = ", ".join(plan.generated_artifact_ids) if plan.generated_artifact_ids else "none"
    run_block = ", ".join(plan.generated_run_ids) if plan.generated_run_ids else "none"
    execution_notes = "\n".join(f"- {item}" for item in plan.execution_notes) or "- none"
    report_path.write_text(
        (
            f"# Gemini Autopilot Report\n\n"
            f"- Created at: {plan.created_at}\n"
            f"- Mode: {plan.mode}\n"
            f"- Provider: {plan.provider}\n"
            f"- Model: {plan.model}\n"
            f"- Safety mode: {plan.safety_mode}\n"
            f"- Recommended direction: {plan.recommended_direction_slug or 'n/a'}\n"
            f"- Applied task IDs: {applied_block}\n\n"
            f"- Executed task IDs: {executed_block}\n"
            f"- Generated artifact IDs: {artifact_block}\n"
            f"- Generated run IDs: {run_block}\n\n"
            "## Summary\n\n"
            f"{plan.summary or 'n/a'}\n\n"
            "## Recommendation\n\n"
            f"{plan.recommended_direction_reason or 'n/a'}\n\n"
            "## Notes\n\n"
            f"{notes}\n\n"
            "## Execution Notes\n\n"
            f"{execution_notes}\n\n"
            "## Task Proposals\n\n"
            f"{tasks_block}\n"
        ),
        encoding="utf-8",
    )
    artifact = repository.create_artifact(
        kind=ArtifactKind.REPORT,
        path=report_path,
        metadata={
            "llm_autopilot": True,
            "mode": plan.mode,
            "provider": plan.provider,
            "model": plan.model,
            "recommended_direction_slug": plan.recommended_direction_slug,
            "applied_task_ids": applied_task_ids,
        },
    )
    return artifact.id, artifact.path


def run_autopilot_cycle(
    repository: LabRepository,
    *,
    max_tasks: int = 3,
    apply: bool = False,
    mode: str = "manual",
) -> LLMAutopilotRun:
    if apply:
        open_tasks = [
            task
            for task in repository.list_tasks()
            if task.status in {"open", "in_progress"}
        ]
        if len(open_tasks) >= max(6, max_tasks * 3):
            execution_batch = execute_supported_tasks(
                repository,
                limit=max(1, min(5, max_tasks)),
            )
            generated_run_ids = list(execution_batch.run_ids)
            chained_run_ids = _queue_maintenance_runs_if_needed(repository)
            for chained_run_id in chained_run_ids:
                if chained_run_id not in generated_run_ids:
                    generated_run_ids.append(chained_run_id)
                    execution_batch.notes.append(f"Queued continuous maintenance run {chained_run_id}.")
            if execution_batch.executed_task_ids or generated_run_ids or execution_batch.artifact_ids:
                return LLMAutopilotRun(
                    provider="system",
                    model="executor-only",
                    safety_mode="review_only",
                    created_at=utc_now(),
                    summary="Executed existing backlog without calling Gemini because enough open tasks already existed.",
                    recommended_direction_slug="",
                    recommended_direction_reason="The lab already had enough executable backlog, so the daemon used that first to reduce cost and clutter.",
                    memory_mode="local_history",
                    notes=["Gemini planning was skipped for this cycle to reduce unnecessary API usage."],
                    task_proposals=[],
                    applied=True,
                    applied_task_ids=[],
                    executed_task_ids=list(execution_batch.executed_task_ids),
                    generated_artifact_ids=list(execution_batch.artifact_ids),
                    generated_run_ids=generated_run_ids,
                    execution_notes=list(execution_batch.notes),
                )

    context = _collect_autopilot_context(repository)
    plan = generate_autopilot_plan(
        baseline=context["baseline"],
        directions=context["directions"],
        tasks=context["tasks"],
        claims=context["claims"],
        sources=context["sources"],
        runs=context["runs"],
        max_tasks=max(1, min(5, max_tasks)),
    )
    applied_task_ids = _apply_task_proposals(repository, plan) if apply else []
    execution_batch = execute_supported_tasks(
        repository,
        limit=max(1, min(5, max_tasks)),
        preferred_task_ids=applied_task_ids if apply else [],
    ) if apply else None
    updated_plan = plan.model_copy(
        update={
            "mode": mode,
            "applied": apply,
            "applied_task_ids": applied_task_ids,
            "executed_task_ids": list(execution_batch.executed_task_ids if execution_batch else []),
            "generated_artifact_ids": list(execution_batch.artifact_ids if execution_batch else []),
            "generated_run_ids": list(execution_batch.run_ids if execution_batch else []),
            "execution_notes": list(execution_batch.notes if execution_batch else []),
        }
    )
    if apply and execution_batch is not None:
        chained_run_ids = _queue_maintenance_runs_if_needed(repository)
        if chained_run_ids:
            merged_run_ids = list(updated_plan.generated_run_ids)
            merged_notes = list(updated_plan.execution_notes)
            for chained_run_id in chained_run_ids:
                if chained_run_id not in merged_run_ids:
                    merged_run_ids.append(chained_run_id)
                    merged_notes.append(f"Queued continuous maintenance run {chained_run_id}.")
            updated_plan = updated_plan.model_copy(
                update={
                    "generated_run_ids": merged_run_ids,
                    "execution_notes": merged_notes,
                }
            )
    should_write_report = updated_plan.provider == "gemini" and (
        mode != "daemon"
        or bool(
            updated_plan.applied_task_ids
            or updated_plan.executed_task_ids
            or updated_plan.generated_artifact_ids
            or updated_plan.generated_run_ids
        )
    )
    artifact_id, artifact_path = (None, None)
    if should_write_report:
        artifact_id, artifact_path = _write_autopilot_report(
            repository,
            plan=updated_plan,
            applied_task_ids=applied_task_ids,
        )
    return updated_plan.model_copy(
        update={
            "report_artifact_id": artifact_id,
            "report_artifact_path": artifact_path,
        }
    )


@dataclass
class _AutopilotRuntime:
    enabled: bool
    interval_seconds: int
    max_tasks: int
    apply: bool
    thread_running: bool = False
    last_run_at: str | None = None
    last_success_at: str | None = None
    last_error: str = ""
    last_summary: str = ""
    last_applied_task_ids: list[str] | None = None
    last_executed_task_ids: list[str] | None = None
    last_generated_artifact_ids: list[str] | None = None
    last_generated_run_ids: list[str] | None = None
    last_recommended_direction_slug: str = ""
    last_report_artifact_id: str | None = None
    last_report_artifact_path: str | None = None
    cycle_count: int = 0


class AutopilotManager:
    def __init__(self, repository: LabRepository, *, enabled: bool, interval_seconds: int, max_tasks: int):
        self._repository = repository
        self._state = _AutopilotRuntime(
            enabled=enabled,
            interval_seconds=max(60, interval_seconds),
            max_tasks=max(1, min(5, max_tasks)),
            apply=True,
            last_applied_task_ids=[],
        )
        self._state_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def status(self) -> LLMAutopilotStatus:
        with self._state_lock:
            return LLMAutopilotStatus(
                enabled=self._state.enabled,
                thread_running=self._state.thread_running,
                interval_seconds=self._state.interval_seconds,
                max_tasks=self._state.max_tasks,
                apply=self._state.apply,
                last_run_at=self._state.last_run_at,
                last_success_at=self._state.last_success_at,
                last_error=self._state.last_error,
                last_summary=self._state.last_summary,
                last_applied_task_ids=list(self._state.last_applied_task_ids or []),
                last_executed_task_ids=list(self._state.last_executed_task_ids or []),
                last_generated_artifact_ids=list(self._state.last_generated_artifact_ids or []),
                last_generated_run_ids=list(self._state.last_generated_run_ids or []),
                last_recommended_direction_slug=self._state.last_recommended_direction_slug,
                last_report_artifact_id=self._state.last_report_artifact_id,
                last_report_artifact_path=self._state.last_report_artifact_path,
                cycle_count=self._state.cycle_count,
            )

    def record_result(self, result: LLMAutopilotRun) -> LLMAutopilotStatus:
        with self._state_lock:
            self._state.last_run_at = result.created_at
            self._state.last_success_at = result.created_at
            self._state.last_error = ""
            self._state.last_summary = result.summary
            self._state.last_applied_task_ids = list(result.applied_task_ids)
            self._state.last_executed_task_ids = list(result.executed_task_ids)
            self._state.last_generated_artifact_ids = list(result.generated_artifact_ids)
            self._state.last_generated_run_ids = list(result.generated_run_ids)
            self._state.last_recommended_direction_slug = result.recommended_direction_slug
            self._state.last_report_artifact_id = result.report_artifact_id
            self._state.last_report_artifact_path = result.report_artifact_path
            self._state.cycle_count += 1
        return self.status()

    def record_error(self, message: str) -> LLMAutopilotStatus:
        with self._state_lock:
            self._state.last_error = message
        return self.status()

    def configure(self, *, enabled: bool, interval_seconds: int, max_tasks: int) -> LLMAutopilotStatus:
        with self._state_lock:
            self._state.enabled = enabled
            self._state.interval_seconds = max(60, interval_seconds)
            self._state.max_tasks = max(1, min(5, max_tasks))
        if enabled:
            self.start()
        else:
            self.stop()
        return self.status()

    def start(self) -> LLMAutopilotStatus:
        should_start = False
        with self._state_lock:
            if not self._state.enabled:
                pass
            elif self._thread is not None and self._thread.is_alive():
                self._state.thread_running = True
            else:
                self._stop_event = threading.Event()
                self._thread = threading.Thread(target=self._loop, name="collatz-gemini-autopilot", daemon=True)
                self._state.thread_running = True
                should_start = True
        if should_start and self._thread is not None:
            self._thread.start()
        return self.status()

    def stop(self) -> LLMAutopilotStatus:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        with self._state_lock:
            self._state.thread_running = False
        return self.status()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            next_sleep_seconds = self.status().interval_seconds
            try:
                result = self.run_now(
                    apply=True,
                    max_tasks=self.status().max_tasks,
                    mode="daemon",
                )
                self.record_result(result)
                next_sleep_seconds = self._next_sleep_seconds(result)
            except Exception as exc:  # pragma: no cover - background runtime behavior
                self.record_error(str(exc))
                next_sleep_seconds = min(120, self.status().interval_seconds)
            if self._stop_event.wait(next_sleep_seconds):
                break
        with self._state_lock:
            self._state.thread_running = False

    def run_now(self, *, apply: bool, max_tasks: int, mode: str) -> LLMAutopilotRun:
        with self._run_lock:
            return run_autopilot_cycle(
                self._repository,
                max_tasks=max_tasks,
                apply=apply,
                mode=mode,
            )

    def _next_sleep_seconds(self, result: LLMAutopilotRun) -> int:
        summary = self._repository.summary()
        if summary.running_run_count == 0 and summary.queued_run_count == 0 and summary.open_task_count > 0:
            return min(self.status().interval_seconds, 60)
        return self.status().interval_seconds
