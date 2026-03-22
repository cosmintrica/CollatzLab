from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

from .repository import LabRepository, utc_now
from .schemas import ArtifactKind, RunStatus, Task, TaskStatus
from .services import collatz_step


@dataclass
class TaskExecutionResult:
    task_id: str
    executed: bool = False
    artifact_ids: list[str] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)
    follow_up_task_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class TaskExecutionBatch:
    executed_task_ids: list[str] = field(default_factory=list)
    artifact_ids: list[str] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)
    follow_up_task_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _task_text(task: Task) -> str:
    return f"{task.title} {task.description}".strip().lower()


def _task_execution_priority(task: Task) -> tuple[int, int, str]:
    text = _task_text(task)
    if task.direction_slug == "verification" and task.kind == "experiment" and any(
        keyword in text for keyword in ("sweep", "queue", "new run", "bounded run")
    ):
        return (0, task.priority, task.id)
    if task.direction_slug == "verification":
        return (1, task.priority, task.id)
    if task.direction_slug == "two-adic-v2":
        return (2, task.priority, task.id)
    if task.direction_slug == "inverse-tree-parity" and task.kind == "experiment":
        return (2, task.priority, task.id)
    if task.direction_slug == "lemma-workspace":
        return (3, task.priority, task.id)
    return (4, task.priority, task.id)


def _record_entries(repository: LabRepository) -> list[dict[str, Any]]:
    runs = [
        run
        for run in repository.list_runs()
        if run.status in {RunStatus.COMPLETED, RunStatus.VALIDATED}
    ]
    entries: list[dict[str, Any]] = []
    for run in runs:
        metrics = run.metrics or {}
        for metric_name in ("max_total_stopping_time", "max_stopping_time", "max_excursion"):
            metric = metrics.get(metric_name) or {}
            raw_seed = metric.get("n")
            raw_value = metric.get("value")
            try:
                seed = int(raw_seed)
                value = int(raw_value)
            except (TypeError, ValueError):
                continue
            if seed < 1:
                continue
            entries.append(
                {
                    "run_id": run.id,
                    "run_name": run.name,
                    "kernel": run.kernel,
                    "hardware": run.hardware,
                    "metric": metric_name,
                    "seed": seed,
                    "value": value,
                }
            )
    return entries


def _parity_signature(seed: int, length: int = 12) -> str:
    current = seed
    bits: list[str] = []
    for _ in range(max(1, length)):
        bits.append("1" if current % 2 else "0")
        if current == 1:
            break
        current = collatz_step(current)
    return "".join(bits)


def _v2(value: int) -> int:
    count = 0
    current = max(1, int(value))
    while current % 2 == 0:
        current //= 2
        count += 1
    return count


def _odd_core(seed: int) -> int:
    current = max(1, int(seed))
    while current % 2 == 0:
        current //= 2
    return current


def _residue_table(unique_seeds: list[int], modulus: int) -> list[dict[str, int]]:
    counts = Counter(seed % modulus for seed in unique_seeds)
    return [
        {"residue": residue, "count": count}
        for residue, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _top_signature_rows(unique_seeds: list[int], limit: int = 8) -> list[dict[str, Any]]:
    counts = Counter(_parity_signature(seed) for seed in unique_seeds)
    return [
        {"signature": signature, "count": count}
        for signature, count in counts.most_common(limit)
    ]


def _render_markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_none_"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _write_report_artifact(
    repository: LabRepository,
    *,
    task: Task,
    title: str,
    body: str,
    claim_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    report_path = repository.settings.reports_dir / f"task-{task.id}.md"
    report_path.write_text(body, encoding="utf-8")
    artifact = repository.create_artifact(
        kind=ArtifactKind.REPORT,
        path=report_path,
        claim_id=claim_id,
        metadata={
            "task_id": task.id,
            "task_title": task.title,
            "result_title": title,
            "executor": "autopilot-task-loop",
            **(metadata or {}),
        },
    )
    return artifact.id


def _write_json_artifact(
    repository: LabRepository,
    *,
    task: Task,
    payload: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> str:
    task_dir = repository.settings.artifacts_dir / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    json_path = task_dir / f"{task.id}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    artifact = repository.create_artifact(
        kind=ArtifactKind.JSON,
        path=json_path,
        metadata={
            "task_id": task.id,
            "executor": "autopilot-task-loop",
            **(metadata or {}),
        },
    )
    return artifact.id


def _ensure_follow_up_task(
    repository: LabRepository,
    *,
    direction_slug: str,
    title: str,
    kind: str,
    description: str,
    owner: str = "integrator",
    priority: int = 2,
) -> str | None:
    normalized_direction = direction_slug.strip().lower()
    normalized_title = title.strip().lower()
    for task in repository.list_tasks():
        if task.direction_slug.strip().lower() == normalized_direction and task.title.strip().lower() == normalized_title:
            return None
    created = repository.create_task(
        direction_slug=direction_slug,
        title=title,
        kind=kind,
        description=description,
        owner=owner,
        priority=priority,
    )
    return created.id


def _build_record_payload(entries: list[dict[str, Any]]) -> dict[str, Any]:
    unique_seeds = sorted({entry["seed"] for entry in entries})
    metric_counts = Counter(entry["metric"] for entry in entries)
    seed_hits = Counter(entry["seed"] for entry in entries)
    unique_run_ids = sorted({entry["run_id"] for entry in entries})
    preview_limit = 12
    seed_preview = unique_seeds[:preview_limit]
    entry_preview = [
        {
            "run_id": entry["run_id"],
            "metric": entry["metric"],
            "seed": entry["seed"],
            "value": entry["value"],
        }
        for entry in entries[:preview_limit]
    ]
    return {
        "schema_version": 2,
        "storage_mode": "summary-first",
        "canonical_source": "rebuild from completed/validated runs stored in SQLite",
        "record_entry_count": len(entries),
        "unique_seed_count": len(unique_seeds),
        "run_count": len(unique_run_ids),
        "run_ids": unique_run_ids,
        "seed_range": {
            "min": unique_seeds[0] if unique_seeds else None,
            "max": unique_seeds[-1] if unique_seeds else None,
        },
        "unique_seed_preview": seed_preview,
        "unique_seed_preview_truncated": len(unique_seeds) > len(seed_preview),
        "metric_counts": dict(metric_counts),
        "top_seed_hits": [
            {"seed": seed, "count": count}
            for seed, count in seed_hits.most_common(12)
        ],
        "mod_8": _residue_table(unique_seeds, 8),
        "mod_16": _residue_table(unique_seeds, 16),
        "mod_32": _residue_table(unique_seeds, 32),
        "parity_signatures": _top_signature_rows(unique_seeds),
        "record_entry_preview": entry_preview,
        "record_entry_preview_truncated": len(entries) > len(entry_preview),
    }


def _replace_managed_notes_section(existing_notes: str, section_title: str, body: str) -> str:
    start_marker = f"<!-- {section_title}:start -->"
    end_marker = f"<!-- {section_title}:end -->"
    existing = existing_notes or ""
    existing = re.sub(
        r"(?ms)^## Task execution update .*?(?=^## Task execution update |^<!-- auto-consolidation:start -->|\Z)",
        "",
        existing,
    ).strip()
    start_index = existing.find(start_marker)
    end_index = existing.find(end_marker)
    preserved = existing
    if start_index != -1 and end_index != -1 and end_index > start_index:
        preserved = (existing[:start_index] + existing[end_index + len(end_marker):]).strip()
    managed_block = f"{start_marker}\n{body.strip()}\n{end_marker}"
    return f"{preserved}\n\n{managed_block}".strip() if preserved else managed_block


def _execute_inverse_tree_task(repository: LabRepository, task: Task) -> TaskExecutionResult:
    entries = _record_entries(repository)
    if not entries:
        return TaskExecutionResult(task_id=task.id, notes=["No completed or validated runs exist yet."])

    payload = _build_record_payload(entries)
    artifact_ids = [
        _write_json_artifact(
            repository,
            task=task,
            payload=payload,
            metadata={"analysis_type": "inverse-tree-parity"},
        )
    ]
    report_body = (
        f"# {task.title}\n\n"
        f"- Task ID: {task.id}\n"
        f"- Direction: {task.direction_slug}\n"
        f"- Executed at: {utc_now()}\n"
        f"- Unique record seeds analyzed: {payload['unique_seed_count']}\n\n"
        "## Why this exists\n\n"
        "This task turns completed-run record seeds into concrete parity and residue summaries.\n"
        "It is structural analysis, not a proof claim.\n\n"
        "## Residues mod 16\n\n"
        f"{_render_markdown_table(payload['mod_16'], ['residue', 'count'])}\n\n"
        "## Residues mod 32\n\n"
        f"{_render_markdown_table(payload['mod_32'], ['residue', 'count'])}\n\n"
        "## Top parity signatures\n\n"
        f"{_render_markdown_table(payload['parity_signatures'], ['signature', 'count'])}\n\n"
        "## Top repeated record seeds\n\n"
        f"{_render_markdown_table(payload['top_seed_hits'], ['seed', 'count'])}\n\n"
        "## Storage note\n\n"
        "The JSON artifact stores a compact summary plus previews only. The canonical raw data stays in SQLite runs.\n"
    )
    artifact_ids.append(
        _write_report_artifact(
            repository,
            task=task,
            title="Inverse-tree parity analysis",
            body=report_body,
            metadata={"analysis_type": "inverse-tree-parity"},
        )
    )

    follow_up_task_ids: list[str] = []
    follow_up_id = _ensure_follow_up_task(
        repository,
        direction_slug="inverse-tree-parity",
        title="Probe residue candidates from current record-seed summary",
        kind="experiment",
        description=(
            f"Use the parity/residue summary from {task.id} to design a bounded modular probe "
            "against the most repeated residue classes."
        ),
        owner="theory-agent",
        priority=2,
    )
    if follow_up_id:
        follow_up_task_ids.append(follow_up_id)

    return TaskExecutionResult(
        task_id=task.id,
        executed=True,
        artifact_ids=artifact_ids,
        follow_up_task_ids=follow_up_task_ids,
        notes=[f"Analyzed {payload['unique_seed_count']} unique record seeds."],
    )


def _execute_two_adic_task(repository: LabRepository, task: Task) -> TaskExecutionResult:
    entries = _record_entries(repository)
    if not entries:
        return TaskExecutionResult(task_id=task.id, notes=["No completed or validated runs exist yet."])

    unique_seeds = sorted({entry["seed"] for entry in entries})
    odd_seeds = sorted({_odd_core(seed) for seed in unique_seeds})
    valuation_rows: list[dict[str, Any]] = []
    v2_counter: Counter[int] = Counter()
    residue_counter: Counter[int] = Counter()
    compressed_preview: list[dict[str, int]] = []
    for odd_seed in odd_seeds:
        lifted = (3 * odd_seed) + 1
        valuation = _v2(lifted)
        odd_image = lifted >> valuation
        valuation_rows.append(
            {
                "odd_seed": odd_seed,
                "v2_3n_plus_1": valuation,
                "odd_image": odd_image,
                "mod_8": odd_seed % 8,
                "mod_16": odd_seed % 16,
            }
        )
        v2_counter[valuation] += 1
        residue_counter[odd_seed % 16] += 1
    compressed_preview = valuation_rows[:16]
    histogram = [
        {"v2": valuation, "count": count}
        for valuation, count in sorted(v2_counter.items(), key=lambda item: (item[0], -item[1]))
    ]
    residue_rows = [
        {"residue": residue, "count": count}
        for residue, count in sorted(residue_counter.items(), key=lambda item: (-item[1], item[0]))
    ]
    payload = {
        "schema_version": 1,
        "analysis_type": "two-adic-v2",
        "unique_seed_count": len(unique_seeds),
        "odd_seed_count": len(odd_seeds),
        "v2_histogram": histogram,
        "odd_seed_mod_16": residue_rows,
        "compressed_preview": compressed_preview,
        "max_v2": max(v2_counter) if v2_counter else 0,
    }
    artifact_ids = [
        _write_json_artifact(
            repository,
            task=task,
            payload=payload,
            metadata={"analysis_type": "two-adic-v2"},
        )
    ]
    report_body = (
        f"# {task.title}\n\n"
        f"- Task ID: {task.id}\n"
        f"- Executed at: {utc_now()}\n"
        f"- Unique seeds considered: {len(unique_seeds)}\n"
        f"- Odd cores considered: {len(odd_seeds)}\n\n"
        "## Why this lane exists\n\n"
        "This lane studies the odd-only compressed step and the valuation v2(3n+1) on odd seeds.\n"
        "It is intended to surface 2-adic regularities, not to replace proof obligations.\n\n"
        "## v2(3n+1) histogram\n\n"
        f"{_render_markdown_table(histogram, ['v2', 'count'])}\n\n"
        "## Odd seeds mod 16\n\n"
        f"{_render_markdown_table(residue_rows, ['residue', 'count'])}\n\n"
        "## Compressed odd-step preview\n\n"
        f"{_render_markdown_table(compressed_preview, ['odd_seed', 'v2_3n_plus_1', 'odd_image', 'mod_8', 'mod_16'])}\n"
    )
    artifact_ids.append(
        _write_report_artifact(
            repository,
            task=task,
            title="2-adic / v2 analysis",
            body=report_body,
            metadata={"analysis_type": "two-adic-v2"},
        )
    )
    follow_up_task_ids: list[str] = []
    follow_up_id = _ensure_follow_up_task(
        repository,
        direction_slug="two-adic-v2",
        title="Test v2(3n+1) clusters against odd-only residues",
        kind="experiment",
        description=(
            f"Take the valuation histogram from {task.id} and probe whether the strongest v2 clusters persist "
            "across odd seeds grouped by residue class."
        ),
        owner="theory-agent",
        priority=2,
    )
    if follow_up_id:
        follow_up_task_ids.append(follow_up_id)

    return TaskExecutionResult(
        task_id=task.id,
        executed=True,
        artifact_ids=artifact_ids,
        follow_up_task_ids=follow_up_task_ids,
        notes=[f"Profiled v2(3n+1) on {len(odd_seeds)} odd cores."],
    )


def _execute_claim_consolidation_task(repository: LabRepository, task: Task) -> TaskExecutionResult:
    entries = _record_entries(repository)
    if not entries:
        return TaskExecutionResult(task_id=task.id, notes=["No completed or validated runs are available to consolidate."])

    claim = None
    # Find the baseline record claim by direction and title rather than
    # relying on a hardcoded ID that may not exist in every database.
    for candidate in repository.list_claims():
        if candidate.direction_slug == "lemma-workspace" and "record" in candidate.title.lower():
            claim = candidate
            break
    if claim is None:
        claims = repository.list_claims()
        for candidate in claims:
            if candidate.direction_slug == "lemma-workspace":
                claim = candidate
                break

    payload = _build_record_payload(entries)
    unique_runs = sorted({entry["run_id"] for entry in entries})
    artifact_ids: list[str] = []
    if claim is not None:
        note_block = (
            "## Auto consolidation snapshot\n\n"
            f"- Last updated at: {utc_now()}\n"
            f"- Last source task: {task.id}\n"
            f"- Consolidated record entries: {len(entries)}\n"
            f"- Unique record seeds: {payload['unique_seed_count']}\n"
            f"- Supporting runs currently linked: {len(unique_runs)}\n"
            f"- Run coverage: {', '.join(unique_runs[:8]) if unique_runs else 'none'}"
            + (f" (+{len(unique_runs) - 8} more)" if len(unique_runs) > 8 else "")
            + "\n"
        )
        merged_notes = _replace_managed_notes_section(claim.notes, "auto-consolidation", note_block)
        repository.update_claim(claim.id, notes=merged_notes, status="supported")
        for run_id in unique_runs:
            repository.link_claim_run(claim_id=claim.id, run_id=run_id, relation="supports")

    report_body = (
        f"# {task.title}\n\n"
        f"- Task ID: {task.id}\n"
        f"- Executed at: {utc_now()}\n"
        f"- Claim target: {claim.id if claim is not None else 'none'}\n"
        f"- Record entries consolidated: {len(entries)}\n"
        f"- Unique record seeds: {payload['unique_seed_count']}\n\n"
        "## Metric counts\n\n"
        f"{_render_markdown_table([{'metric': key, 'count': value} for key, value in payload['metric_counts'].items()], ['metric', 'count'])}\n\n"
        "## Top repeated record seeds\n\n"
        f"{_render_markdown_table(payload['top_seed_hits'], ['seed', 'count'])}\n\n"
        "## Residues mod 16\n\n"
        f"{_render_markdown_table(payload['mod_16'], ['residue', 'count'])}\n\n"
        "## Storage note\n\n"
        "This report consolidates canonical run data. The paired JSON artifact is summary-first, not a raw dump.\n"
    )
    artifact_ids.append(
        _write_report_artifact(
            repository,
            task=task,
            title="Claim consolidation report",
            body=report_body,
            claim_id=claim.id if claim is not None else None,
            metadata={
                "analysis_type": "claim-consolidation",
                "claim_id": claim.id if claim is not None else None,
                "linked_run_ids": unique_runs,
            },
        )
    )

    follow_up_task_ids: list[str] = []
    follow_up_id = _ensure_follow_up_task(
        repository,
        direction_slug="inverse-tree-parity",
        title="Analyze parity vectors for consolidated claim seeds",
        kind="analysis",
        description=(
            f"Take the consolidated record seeds refreshed by {task.id} and compare their parity signatures, "
            "residue classes, and recurrence counts."
        ),
        owner="theory-agent",
        priority=2,
    )
    if follow_up_id:
        follow_up_task_ids.append(follow_up_id)

    return TaskExecutionResult(
        task_id=task.id,
        executed=True,
        artifact_ids=artifact_ids,
        follow_up_task_ids=follow_up_task_ids,
        notes=[f"Consolidated {len(entries)} record entries into {claim.id if claim is not None else 'a task report'}."],
    )


def _execute_indirect_transform_task(repository: LabRepository, task: Task) -> TaskExecutionResult:
    claims = repository.list_claims()
    sources = repository.list_sources()
    report_body = (
        f"# {task.title}\n\n"
        f"- Task ID: {task.id}\n"
        f"- Executed at: {utc_now()}\n"
        f"- Existing claims in lab: {len(claims)}\n"
        f"- Existing sources in lab: {len(sources)}\n\n"
        "## Transform lane template\n\n"
        "1. Define the transform explicitly.\n"
        "2. State what property it preserves.\n"
        "3. State what property it weakens or loses.\n"
        "4. Explain how success would help with descent, cycles, or boundedness.\n"
        "5. Define the first deterministic falsifier for the proposed lane.\n\n"
        "## Guardrails\n\n"
        "- Do not treat analogy as proof.\n"
        "- Do not assume a surrogate map preserves all orbit properties.\n"
        "- Require one bounded claim that can be tested or refuted next.\n"
    )
    artifact_id = _write_report_artifact(
        repository,
        task=task,
        title="Indirect transform workspace",
        body=report_body,
        metadata={"analysis_type": "indirect-transform-workspace"},
    )
    follow_up_task_ids: list[str] = []
    follow_up_id = _ensure_follow_up_task(
        repository,
        direction_slug="lemma-workspace",
        title="Specify one invariant-preserving surrogate transform",
        kind="theory",
        description=(
            f"Use the workspace report from {task.id} to define one concrete transform candidate, "
            "its preserved quantity, and the first deterministic falsifier."
        ),
        owner="theory-agent",
        priority=2,
    )
    if follow_up_id:
        follow_up_task_ids.append(follow_up_id)
    return TaskExecutionResult(
        task_id=task.id,
        executed=True,
        artifact_ids=[artifact_id],
        follow_up_task_ids=follow_up_task_ids,
        notes=["Created an indirect-transform workspace with explicit proof obligations."],
    )


def _execute_source_review_task(repository: LabRepository, task: Task) -> TaskExecutionResult:
    text = _task_text(task)
    sources = repository.list_sources()
    matched_sources = [
        source
        for source in sources
        if any(token in f"{source.title} {source.authors} {source.url}".lower() for token in text.split())
    ]
    report_body = (
        f"# {task.title}\n\n"
        f"- Task ID: {task.id}\n"
        f"- Executed at: {utc_now()}\n"
        f"- Registered sources inspected: {len(sources)}\n"
        f"- Matching sources already in lab: {len(matched_sources)}\n\n"
        "## Result\n\n"
        + (
            "No matching registered source was found in the local source registry yet.\n"
            "This task produced a review-prep note instead of a final source review.\n"
            if not matched_sources
            else "The following registered sources match this review task and are ready for manual or assisted review.\n"
        )
        + "\n\n"
        + (
            _render_markdown_table(
                [
                    {
                        "id": source.id,
                        "title": source.title,
                        "status": source.review_status.value,
                        "claim_type": source.claim_type.value,
                    }
                    for source in matched_sources
                ],
                ["id", "title", "status", "claim_type"],
            )
            if matched_sources
            else "## Missing registration next step\n\n"
            "Register the target paper/blog/source in Sources, then rerun a focused review task against that explicit entry.\n"
        )
    )
    artifact_id = _write_report_artifact(
        repository,
        task=task,
        title="Source review preparation",
        body=report_body,
        metadata={"analysis_type": "source-review-prep", "matched_source_ids": [source.id for source in matched_sources]},
    )
    follow_up_task_ids: list[str] = []
    if not matched_sources:
        follow_up_id = _ensure_follow_up_task(
            repository,
            direction_slug="lemma-workspace",
            title="Register missing source for pending review",
            kind="source-review",
            description=(
                f"Add the explicit source entry needed for task {task.id}, then run a formal review against that registered source."
            ),
            owner="integrator",
            priority=2,
        )
        if follow_up_id:
            follow_up_task_ids.append(follow_up_id)
    return TaskExecutionResult(
        task_id=task.id,
        executed=True,
        artifact_ids=[artifact_id],
        follow_up_task_ids=follow_up_task_ids,
        notes=[
            "Prepared a source-review artifact." if matched_sources else "Prepared a source-review intake note and follow-up registration task."
        ],
    )


def _execute_verification_task(repository: LabRepository, task: Task) -> TaskExecutionResult:
    text = _task_text(task)
    if task.kind == "experiment" and any(keyword in text for keyword in ("sweep", "queue", "new run", "bounded run")):
        existing_runs = repository.list_runs()
        next_start = max((run.range_end for run in existing_runs), default=0) + 1
        next_end = next_start + 99_999_999
        run = repository.create_run(
            direction_slug="verification",
            name=f"autopilot-{task.id.lower()}",
            range_start=next_start,
            range_end=next_end,
            kernel="cpu-parallel",
            hardware="cpu",
            owner="gemini-autopilot",
        )
        artifact_id = _write_report_artifact(
            repository,
            task=task,
            title="Queued verification run",
            body=(
                f"# {task.title}\n\n"
                f"- Task ID: {task.id}\n"
                f"- Queued run: {run.id}\n"
                f"- Range: {next_start}-{next_end}\n"
                f"- Kernel: {run.kernel}\n"
                f"- Hardware: {run.hardware}\n"
            ),
            metadata={"analysis_type": "queued-run", "run_id": run.id},
        )
        return TaskExecutionResult(
            task_id=task.id,
            executed=True,
            artifact_ids=[artifact_id],
            run_ids=[run.id],
            notes=[f"Queued bounded verification run {run.id}."],
        )

    entries = _record_entries(repository)
    if not entries:
        return TaskExecutionResult(task_id=task.id, notes=["No completed or validated runs exist yet."])

    metric_focus = "max_excursion" if "excursion" in text else "max_total_stopping_time"
    focused_rows = [
        entry
        for entry in entries
        if entry["metric"] == metric_focus
    ]
    report_body = (
        f"# {task.title}\n\n"
        f"- Task ID: {task.id}\n"
        f"- Executed at: {utc_now()}\n"
        f"- Focus metric: {metric_focus}\n"
        f"- Rows analyzed: {len(focused_rows)}\n\n"
        "## Focus rows\n\n"
        f"{_render_markdown_table(focused_rows[:20], ['run_id', 'kernel', 'hardware', 'seed', 'value'])}\n"
    )
    artifact_id = _write_report_artifact(
        repository,
        task=task,
        title="Verification cross-run analysis",
        body=report_body,
        metadata={"analysis_type": "verification-analysis", "metric_focus": metric_focus},
    )
    return TaskExecutionResult(
        task_id=task.id,
        executed=True,
        artifact_ids=[artifact_id],
        notes=[f"Analyzed {len(focused_rows)} {metric_focus} rows across completed runs."],
    )


def execute_task(repository: LabRepository, task_id: str) -> TaskExecutionResult:
    task = repository.get_task(task_id)
    if task.status == TaskStatus.DONE:
        return TaskExecutionResult(task_id=task.id, notes=["Task is already done."])
    if task.status == TaskStatus.FROZEN:
        return TaskExecutionResult(task_id=task.id, notes=["Task is frozen and will not be auto-executed."])

    repository.update_task(task.id, status=TaskStatus.IN_PROGRESS)
    text = _task_text(task)

    try:
        if task.direction_slug == "inverse-tree-parity":
            result = _execute_inverse_tree_task(repository, task)
        elif task.direction_slug == "two-adic-v2":
            result = _execute_two_adic_task(repository, task)
        elif task.kind == "source-review" or ("formal review" in text and task.direction_slug == "lemma-workspace"):
            result = _execute_source_review_task(repository, task)
        elif any(keyword in text for keyword in ("indirect transform", "surrogate", "bijection")):
            result = _execute_indirect_transform_task(repository, task)
        elif task.direction_slug == "lemma-workspace":
            result = _execute_claim_consolidation_task(repository, task)
        elif task.direction_slug == "verification":
            result = _execute_verification_task(repository, task)
        else:
            result = TaskExecutionResult(task_id=task.id, notes=["No automatic executor exists for this task yet."])
    except Exception as exc:
        repository.update_task(task.id, status=TaskStatus.OPEN)
        return TaskExecutionResult(task_id=task.id, notes=[f"Execution failed: {exc}"])

    repository.update_task(
        task.id,
        status=TaskStatus.DONE if result.executed else TaskStatus.OPEN,
    )
    return result


def execute_supported_tasks(
    repository: LabRepository,
    *,
    limit: int = 3,
    preferred_task_ids: list[str] | None = None,
) -> TaskExecutionBatch:
    preferred_task_ids = preferred_task_ids or []
    open_tasks = [task for task in repository.list_tasks() if task.status == TaskStatus.OPEN]
    tasks_by_id = {task.id: task for task in open_tasks}
    ranked_tasks = sorted(open_tasks, key=_task_execution_priority)
    ordered_ids: list[str] = []
    seen: set[str] = set()
    for task_id in preferred_task_ids:
        if task_id in tasks_by_id and task_id not in seen:
            ordered_ids.append(task_id)
            seen.add(task_id)
    for task in ranked_tasks:
        if task.id in seen:
            continue
        ordered_ids.append(task.id)
        seen.add(task.id)

    batch = TaskExecutionBatch()
    for task_id in ordered_ids:
        if len(batch.executed_task_ids) >= max(1, limit):
            break
        result = execute_task(repository, task_id)
        batch.notes.extend(result.notes)
        if not result.executed:
            continue
        batch.executed_task_ids.append(result.task_id)
        batch.artifact_ids.extend(result.artifact_ids)
        batch.run_ids.extend(result.run_ids)
        batch.follow_up_task_ids.extend(result.follow_up_task_ids)
    return batch
