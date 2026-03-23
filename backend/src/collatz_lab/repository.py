from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Settings
from .database import SCHEMA, connect
from .schemas import (
    Artifact,
    ArtifactKind,
    Claim,
    ClaimRunLink,
    ComputeProfile,
    ConsensusBaseline,
    ConsensusBaselineItem,
    DashboardSummary,
    Direction,
    DirectionReview,
    DirectionStatus,
    FallacyTagInfo,
    HardwareCapability,
    MapVariant,
    Run,
    RunStatus,
    VALID_RUN_TRANSITIONS,
    ReviewRubric,
    Source,
    SourceClaimType,
    SourceReviewEvent,
    SourceReviewMode,
    SourceStatus,
    SourceType,
    Task,
    TaskStatus,
    Worker,
    WorkerStatus,
)


DEFAULT_DIRECTIONS: tuple[dict[str, str], ...] = (
    {
        "slug": "verification",
        "title": "Verification",
        "description": "High-volume interval verification, record tracking, and kernel comparisons.",
        "owner": "compute-agent",
        "success_criteria": "Find reproducible invariants or non-trivial compression in verified intervals.",
        "abandon_criteria": "No reproducible signal after repeated validated runs and independent rechecks.",
    },
    {
        "slug": "inverse-tree-parity",
        "title": "Inverse Tree Parity",
        "description": "Explore odd-node reverse trees, residue classes, and parity-vector filters.",
        "owner": "theory-agent",
        "success_criteria": "Discover modular or parity constraints that persist across extended ranges.",
        "abandon_criteria": "Candidate filters break under wider search or independent implementations.",
    },
    {
        "slug": "lemma-workspace",
        "title": "Lemma Workspace",
        "description": "Track claims, dependencies, counterexamples, and intermediate proof drafts.",
        "owner": "integrator",
        "success_criteria": "Promote evidence-backed claims toward formalization without manual rewrites.",
        "abandon_criteria": "Claims repeatedly fail against linked evidence or never accumulate support.",
    },
    {
        "slug": "two-adic-v2",
        "title": "2-adic / v2 Explorer",
        "description": "Study v2(3n+1), odd-only compression, residue lifting, and 2-adic structural signals.",
        "owner": "theory-agent",
        "success_criteria": "Find stable v2 or residue patterns that survive extension and suggest a concrete next probe.",
        "abandon_criteria": "Candidate 2-adic regularities collapse under wider record-seed or odd-only testing.",
    },
    {
        "slug": "hypothesis-sandbox",
        "title": "Hypothesis Sandbox",
        "description": (
            "Experimental track for generating and testing novel mathematical hypotheses. "
            "Produces hypotheses, transformations, and probes/falsifiers — never promotes "
            "directly to supported/validated claims. All results remain isolated from the "
            "verification pipeline."
        ),
        "owner": "theory-agent",
        "success_criteria": "Generate falsifiable hypotheses with clear test criteria and accumulate plausible patterns worth formal investigation.",
        "abandon_criteria": "Hypotheses consistently falsified on first test or produce only noise without actionable patterns.",
    },
)


CONSENSUS_BASELINE = ConsensusBaseline(
    problem_status="open",
    checked_as_of="2026-03-21",
    verified_up_to="2^71 (~2.36 x 10^21)",
    note="Treat computational verification, partial theorems, and proof attempts as distinct evidence classes.",
    items=[
        ConsensusBaselineItem(
            title="Open problem consensus",
            detail="The Collatz conjecture remains unsolved in standard references as of March 21, 2026.",
            source_url="https://en.wikipedia.org/wiki/Collatz_conjecture",
        ),
        ConsensusBaselineItem(
            title="Computational verification",
            detail="Verification has been extended to 2^71, which is strong evidence but not a proof.",
            source_url="https://www.fit.vut.cz/research/publication/c197809/.en",
        ),
        ConsensusBaselineItem(
            title="Partial theorem",
            detail="Tao proved an 'almost all' result, which is legitimate progress but not a full proof.",
            source_url="https://terrytao.wordpress.com/2019/09/10/almost-all-collatz-orbits-attain-almost-bounded-values/",
        ),
    ],
)


FALLACY_TAG_CATALOG: tuple[FallacyTagInfo, ...] = (
    FallacyTagInfo(
        tag="empirical-not-proof",
        label="Empirical is not proof",
        description="Large computational checks are evidence, not a universal proof.",
    ),
    FallacyTagInfo(
        tag="almost-all-not-all",
        label="Almost all is not all",
        description="Density or almost-everywhere results do not settle the universal conjecture.",
    ),
    FallacyTagInfo(
        tag="circular-descent",
        label="Circular descent",
        description="The descent argument assumes the very global property it claims to prove.",
    ),
    FallacyTagInfo(
        tag="unchecked-generalization",
        label="Unchecked generalization",
        description="A local pattern is promoted to all integers without a valid universal step.",
    ),
    FallacyTagInfo(
        tag="reverse-tree-gap",
        label="Reverse tree gap",
        description="Connectivity or inverse-tree coverage is asserted without closing the forward implication gap.",
    ),
    FallacyTagInfo(
        tag="publishing-does-not-imply-validity",
        label="Publication is not validation",
        description="Posting or publishing a manuscript does not make a proof correct.",
    ),
    FallacyTagInfo(
        tag="variant-confusion",
        label="Map variant confusion",
        description="The source mixes standard, shortcut, odd-only, or inverse-tree variants without proving equivalence.",
    ),
    FallacyTagInfo(
        tag="proof-by-large-search",
        label="Proof by large search",
        description="Finite verification is treated as if it solved the infinite problem.",
    ),
    FallacyTagInfo(
        tag="statistical-leap",
        label="Statistical leap",
        description="Probabilistic or average-case language is used to conclude a deterministic theorem.",
    ),
)

KNOWN_FALLACY_TAGS = {item.tag for item in FALLACY_TAG_CATALOG}


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_within_workspace(workspace_root: Path, candidate: Path) -> Path:
    resolved_root = workspace_root.resolve()
    resolved_candidate = candidate.resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise KeyError(f"Artifact path escapes workspace: {candidate}") from exc
    return resolved_candidate


def normalize_fallacy_tags(tags: list[str] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in tags or []:
        tag = str(raw or "").strip().lower()
        if not tag or tag in seen:
            continue
        if tag not in KNOWN_FALLACY_TAGS:
            raise ValueError(f"Unknown fallacy tag: {tag}")
        normalized.append(tag)
        seen.add(tag)
    return normalized


class LabRepository:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.ensure_directories()

    def init(self) -> None:
        with connect(str(self.settings.db_path)) as conn:
            conn.executescript(SCHEMA)
            self._migrate_schema(conn)
            conn.execute(
                "INSERT OR IGNORE INTO sequences(name, value) VALUES (?, ?)",
                ("global", 0),
            )
            conn.commit()
        self.seed_runtime_settings()
        self.seed_default_directions()
        self.seed_direction_notes()
        self.seed_initial_tasks()

    def _migrate_schema(self, conn) -> None:
        source_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(sources)").fetchall()
        }
        if source_columns and "map_variant" not in source_columns:
            conn.execute(
                "ALTER TABLE sources ADD COLUMN map_variant TEXT NOT NULL DEFAULT 'unspecified'"
            )

    def seed_default_directions(self) -> None:
        timestamp = utc_now()
        with connect(str(self.settings.db_path)) as conn:
            existing = {
                row["slug"]
                for row in conn.execute("SELECT slug FROM directions").fetchall()
            }
            for spec in DEFAULT_DIRECTIONS:
                if spec["slug"] in existing:
                    continue
                conn.execute(
                    """
                    INSERT OR IGNORE INTO directions(
                      id, slug, title, description, owner, status, score,
                      success_criteria, abandon_criteria, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.next_id(conn),
                        spec["slug"],
                        spec["title"],
                        spec["description"],
                        spec["owner"],
                        DirectionStatus.ACTIVE.value,
                        0.0,
                        spec["success_criteria"],
                        spec["abandon_criteria"],
                        timestamp,
                        timestamp,
                    ),
                )
            conn.commit()

    def seed_direction_notes(self) -> None:
        templates = {
            "verification.md": "# Verification\n\n- Goal: build validated interval sweeps and track records.\n- Notes:\n",
            "inverse-tree-parity.md": "# Inverse Tree Parity\n\n- Goal: study reverse trees and parity constraints.\n- Notes:\n",
            "lemma-workspace.md": "# Lemma Workspace\n\n- Goal: record claims, dependencies, and counterexamples.\n- Notes:\n",
            "two-adic-v2.md": "# 2-adic / v2 Explorer\n\n- Goal: study odd-only compression, v2(3n+1), and residue lifting.\n- Notes:\n",
        }
        for name, content in templates.items():
            path = self.settings.research_dir / "directions" / name
            if not path.exists():
                path.write_text(content, encoding="utf-8")

    def seed_runtime_settings(self) -> None:
        timestamp = utc_now()
        with connect(str(self.settings.db_path)) as conn:
            existing = conn.execute(
                "SELECT value_json FROM runtime_settings WHERE key = ?",
                ("compute_profile",),
            ).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO runtime_settings(key, value_json, updated_at) VALUES (?, ?, ?)",
                    (
                        "compute_profile",
                        json.dumps(
                            {
                                "system_percent": 100,
                                "cpu_percent": 100,
                                "gpu_percent": 100,
                            }
                        ),
                        timestamp,
                    ),
                )
                conn.commit()

    def seed_initial_tasks(self) -> None:
        seeds = (
            {
                "direction_slug": "two-adic-v2",
                "title": "Profile v2(3n+1) on current record seeds",
                "kind": "analysis",
                "description": "Summarize v2(3n+1) valuations, odd-only compression, and residue classes for current record seeds.",
                "owner": "theory-agent",
                "priority": 2,
            },
        )
        with connect(str(self.settings.db_path)) as conn:
            existing_signatures = {
                (row["direction_slug"].strip().lower(), row["title"].strip().lower())
                for row in conn.execute("SELECT direction_slug, title FROM tasks").fetchall()
            }
            timestamp = utc_now()
            for spec in seeds:
                signature = (spec["direction_slug"].strip().lower(), spec["title"].strip().lower())
                if signature in existing_signatures:
                    continue
                conn.execute(
                    """
                    INSERT INTO tasks(
                      id, direction_slug, title, kind, description,
                      owner, status, priority, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.next_id(conn),
                        spec["direction_slug"],
                        spec["title"],
                        spec["kind"],
                        spec["description"],
                        spec["owner"],
                        TaskStatus.OPEN.value,
                        spec["priority"],
                        timestamp,
                        timestamp,
                    ),
                )
                existing_signatures.add(signature)
            conn.commit()

    def next_id(self, conn=None) -> str:
        owns_connection = conn is None
        connection = conn or connect(str(self.settings.db_path))
        try:
            connection.execute(
                "INSERT OR IGNORE INTO sequences(name, value) VALUES (?, ?)",
                ("global", 0),
            )
            row = connection.execute(
                """
                UPDATE sequences
                SET value = value + 1
                WHERE name = ?
                RETURNING value
                """,
                ("global",),
            ).fetchone()
            if row is None:
                raise RuntimeError("Failed to reserve a new Collatz Lab ID.")
            next_value = int(row["value"])
            if owns_connection:
                connection.commit()
            return f"COL-{next_value:04d}"
        finally:
            if owns_connection:
                connection.close()

    def create_task(
        self,
        *,
        direction_slug: str,
        title: str,
        kind: str,
        description: str,
        owner: str = "integrator",
        priority: int = 2,
    ) -> Task:
        timestamp = utc_now()
        with connect(str(self.settings.db_path)) as conn:
            task_id = self.next_id(conn)
            conn.execute(
                """
                INSERT INTO tasks(
                  id, direction_slug, title, kind, description,
                  owner, status, priority, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    direction_slug,
                    title,
                    kind,
                    description,
                    owner,
                    TaskStatus.OPEN.value,
                    priority,
                    timestamp,
                    timestamp,
                ),
            )
            conn.commit()
        return self.get_task(task_id)

    def update_task(
        self,
        task_id: str,
        *,
        status: TaskStatus | str | None = None,
        owner: str | None = None,
        description: str | None = None,
        priority: int | None = None,
    ) -> Task:
        fields = ["updated_at = ?"]
        values: list[Any] = [utc_now()]
        if status is not None:
            fields.append("status = ?")
            values.append(status.value if isinstance(status, TaskStatus) else status)
        if owner is not None:
            fields.append("owner = ?")
            values.append(owner)
        if description is not None:
            fields.append("description = ?")
            values.append(description)
        if priority is not None:
            fields.append("priority = ?")
            values.append(priority)
        values.append(task_id)
        with connect(str(self.settings.db_path)) as conn:
            conn.execute(
                f"UPDATE tasks SET {', '.join(fields)} WHERE id = ?",
                values,
            )
            conn.commit()
        return self.get_task(task_id)

    def create_run(
        self,
        *,
        direction_slug: str,
        name: str,
        range_start: int,
        range_end: int,
        kernel: str = "cpu-direct",
        owner: str = "compute-agent",
        code_version: str = "workspace",
        hardware: str = "cpu",
    ) -> Run:
        if not isinstance(range_start, int) or range_start < 1:
            raise ValueError(f"range_start must be a positive integer, got {range_start!r}")
        if not isinstance(range_end, int) or range_end < range_start:
            raise ValueError(
                f"range_end must be an integer >= range_start ({range_start}), got {range_end!r}"
            )
        timestamp = utc_now()
        with connect(str(self.settings.db_path)) as conn:
            run_id = self.next_id(conn)
            conn.execute(
                """
                INSERT INTO runs(
                  id, direction_slug, name, status, range_start, range_end,
                  kernel, owner, checkpoint_json, metrics_json, summary,
                  code_version, hardware, checksum, created_at, started_at, finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    direction_slug,
                    name,
                    RunStatus.QUEUED.value,
                    range_start,
                    range_end,
                    kernel,
                    owner,
                    json.dumps({}),
                    json.dumps({}),
                    "",
                    code_version,
                    hardware,
                    "",
                    timestamp,
                    None,
                    None,
                ),
            )
            conn.commit()
        return self.get_run(run_id)

    def register_worker(
        self,
        *,
        name: str,
        role: str,
        hardware: str,
        capabilities: list[dict[str, Any]],
    ) -> Worker:
        timestamp = utc_now()
        with connect(str(self.settings.db_path)) as conn:
            existing = conn.execute(
                """
                SELECT id
                FROM workers
                WHERE name = ?
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 1
                """,
                (name,),
            ).fetchone()
            if existing is None:
                worker_id = self.next_id(conn)
                conn.execute(
                    """
                    INSERT INTO workers(
                      id, name, role, status, hardware, capabilities_json,
                      current_run_id, created_at, updated_at, last_heartbeat_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        worker_id,
                        name,
                        role,
                        WorkerStatus.IDLE.value,
                        hardware,
                        json.dumps(capabilities),
                        None,
                        timestamp,
                        timestamp,
                        timestamp,
                    ),
                )
            else:
                worker_id = existing["id"]
                conn.execute(
                    """
                    UPDATE workers
                    SET role = ?, status = ?, hardware = ?, capabilities_json = ?,
                        current_run_id = ?, updated_at = ?, last_heartbeat_at = ?
                    WHERE id = ?
                    """,
                    (
                        role,
                        WorkerStatus.IDLE.value,
                        hardware,
                        json.dumps(capabilities),
                        None,
                        timestamp,
                        timestamp,
                        worker_id,
                    ),
                )
            conn.commit()
        self.requeue_orphaned_runs()
        return self.get_worker(worker_id)

    def update_run(
        self,
        run_id: str,
        *,
        status: RunStatus | None = None,
        checkpoint: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        summary: str | None = None,
        checksum: str | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
    ) -> Run:
        fields: list[str] = []
        values: list[Any] = []
        if status is not None:
            current_run = self.get_run(run_id)
            current_status = RunStatus(current_run.status)
            if status != current_status:
                allowed = VALID_RUN_TRANSITIONS.get(current_status, frozenset())
                if status not in allowed:
                    raise ValueError(
                        f"Invalid run status transition: {current_status.value} → {status.value} "
                        f"(allowed: {', '.join(s.value for s in allowed) or 'none'})"
                    )
            fields.append("status = ?")
            values.append(status.value)
        if checkpoint is not None:
            fields.append("checkpoint_json = ?")
            values.append(json.dumps(checkpoint))
        if metrics is not None:
            fields.append("metrics_json = ?")
            values.append(json.dumps(metrics))
        if summary is not None:
            fields.append("summary = ?")
            values.append(summary)
        if checksum is not None:
            fields.append("checksum = ?")
            values.append(checksum)
        if started_at is not None:
            fields.append("started_at = ?")
            values.append(started_at)
        if finished_at is not None:
            fields.append("finished_at = ?")
            values.append(finished_at)
        if not fields:
            return self.get_run(run_id)
        values.append(run_id)
        with connect(str(self.settings.db_path)) as conn:
            conn.execute(f"UPDATE runs SET {', '.join(fields)} WHERE id = ?", values)
            conn.commit()
        return self.get_run(run_id)

    def delete_run(self, run_id: str) -> None:
        with connect(str(self.settings.db_path)) as conn:
            conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            conn.commit()

    def update_worker(
        self,
        worker_id: str,
        *,
        status: str | WorkerStatus | None = None,
        current_run_id: str | None = None,
        capabilities: list[dict[str, Any]] | None = None,
        heartbeat: bool = True,
    ) -> Worker:
        fields: list[str] = ["updated_at = ?"]
        values: list[Any] = [utc_now()]
        if status is not None:
            fields.append("status = ?")
            values.append(status.value if isinstance(status, WorkerStatus) else status)
        fields.append("current_run_id = ?")
        values.append(current_run_id)
        if capabilities is not None:
            fields.append("capabilities_json = ?")
            values.append(json.dumps(capabilities))
        if heartbeat:
            fields.append("last_heartbeat_at = ?")
            values.append(utc_now())
        values.append(worker_id)
        with connect(str(self.settings.db_path)) as conn:
            conn.execute(
                f"UPDATE workers SET {', '.join(fields)} WHERE id = ?",
                values,
            )
            conn.commit()
        return self.get_worker(worker_id)

    def requeue_orphaned_runs(self) -> int:
        recovery_note = "Recovered after worker restart; queued to resume from the last checkpoint."
        with connect(str(self.settings.db_path)) as conn:
            rows = conn.execute(
                """
                SELECT id, summary
                FROM runs
                WHERE status = ?
                  AND id NOT IN (
                    SELECT current_run_id
                    FROM workers
                    WHERE current_run_id IS NOT NULL AND status = ?
                  )
                """,
                (RunStatus.RUNNING.value, WorkerStatus.RUNNING.value),
            ).fetchall()
            if not rows:
                return 0
            for row in rows:
                summary = (row["summary"] or "").strip()
                next_summary = recovery_note if not summary else f"{summary} {recovery_note}"
                conn.execute(
                    """
                    UPDATE runs
                    SET status = ?, summary = ?
                    WHERE id = ?
                    """,
                    (RunStatus.QUEUED.value, next_summary, row["id"]),
                )
            conn.commit()
        return len(rows)

    def create_claim(
        self,
        *,
        direction_slug: str,
        title: str,
        statement: str,
        owner: str = "theory-agent",
        dependencies: list[str] | None = None,
        notes: str = "",
    ) -> Claim:
        timestamp = utc_now()
        dependencies = dependencies or []
        with connect(str(self.settings.db_path)) as conn:
            claim_id = self.next_id(conn)
            conn.execute(
                """
                INSERT INTO claims(
                  id, direction_slug, title, statement, status, owner,
                  dependencies_json, notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    claim_id,
                    direction_slug,
                    title,
                    statement,
                    "idea",
                    owner,
                    json.dumps(dependencies),
                    notes,
                    timestamp,
                    timestamp,
                ),
            )
            conn.commit()

        note_path = self.settings.research_dir / "claims" / f"{claim_id}.md"
        note_path.write_text(
            f"# {title}\n\n- Claim ID: {claim_id}\n- Direction: {direction_slug}\n- Status: idea\n\n## Statement\n\n{statement}\n\n## Dependencies\n\n{json.dumps(dependencies)}\n\n## Notes\n\n{notes}\n",
            encoding="utf-8",
        )
        self.create_artifact(
            kind=ArtifactKind.NOTE,
            path=note_path,
            claim_id=claim_id,
            metadata={"title": title, "direction": direction_slug},
        )
        return self.get_claim(claim_id)

    def update_claim_status(self, claim_id: str, status: str) -> Claim:
        with connect(str(self.settings.db_path)) as conn:
            conn.execute(
                "UPDATE claims SET status = ?, updated_at = ? WHERE id = ?",
                (status, utc_now(), claim_id),
            )
            conn.commit()
        return self.get_claim(claim_id)

    def update_claim(
        self,
        claim_id: str,
        *,
        statement: str | None = None,
        notes: str | None = None,
        status: str | None = None,
    ) -> Claim:
        fields = ["updated_at = ?"]
        values: list[Any] = [utc_now()]
        if statement is not None:
            fields.append("statement = ?")
            values.append(statement)
        if notes is not None:
            fields.append("notes = ?")
            values.append(notes)
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        values.append(claim_id)
        with connect(str(self.settings.db_path)) as conn:
            conn.execute(
                f"UPDATE claims SET {', '.join(fields)} WHERE id = ?",
                values,
            )
            conn.commit()
        return self.get_claim(claim_id)

    def list_fallacy_tags(self) -> list[FallacyTagInfo]:
        return list(FALLACY_TAG_CATALOG)

    def _write_source_note(self, source: Source) -> Path:
        note_path = self.settings.research_dir / "sources" / f"{source.id}.md"
        note_path.write_text(
            (
                f"# {source.title}\n\n"
                f"- Source ID: {source.id}\n"
                f"- Direction: {source.direction_slug}\n"
                f"- Type: {source.source_type.value}\n"
                f"- Claim Type: {source.claim_type.value}\n"
                f"- Review Status: {source.review_status.value}\n"
                f"- Map Variant: {source.map_variant.value}\n"
                f"- URL: {source.url or 'n/a'}\n"
                f"- Authors: {source.authors or 'n/a'}\n"
                f"- Year: {source.year or 'n/a'}\n\n"
                "## Summary\n\n"
                f"{source.summary or 'n/a'}\n\n"
                "## Review Rubric\n\n"
                f"{json.dumps(source.rubric.model_dump(), indent=2)}\n\n"
                "## Fallacy Tags\n\n"
                f"{json.dumps(source.fallacy_tags)}\n\n"
                "## Notes\n\n"
                f"{source.notes or 'n/a'}\n"
            ),
            encoding="utf-8",
        )
        return note_path

    def _write_source_review_snapshot(self, source: Source) -> Path:
        timestamp_slug = source.updated_at.replace(":", "-").replace("+", "_")
        review_path = self.settings.artifacts_dir / "reviews" / f"{source.id}-{timestamp_slug}.md"
        review_path.write_text(
            (
                f"# Review Snapshot for {source.title}\n\n"
                f"- Source ID: {source.id}\n"
                f"- Review Status: {source.review_status.value}\n"
                f"- Claim Type: {source.claim_type.value}\n"
                f"- Map Variant: {source.map_variant.value}\n"
                f"- Fallacy Tags: {', '.join(source.fallacy_tags) if source.fallacy_tags else 'none'}\n\n"
                "## Summary\n\n"
                f"{source.summary or 'n/a'}\n\n"
                "## Rubric\n\n"
                f"{json.dumps(source.rubric.model_dump(), indent=2)}\n\n"
                "## Notes\n\n"
                f"{source.notes or 'n/a'}\n"
            ),
            encoding="utf-8",
        )
        return review_path

    def _record_source_review_event(
        self,
        *,
        source_id: str,
        reviewer: str,
        mode: SourceReviewMode | str,
        review_status: SourceStatus | str | None = None,
        map_variant: MapVariant | str | None = None,
        summary: str = "",
        notes: str = "",
        fallacy_tags: list[str] | None = None,
        rubric: ReviewRubric | None = None,
        llm_provider: str = "",
        llm_model: str = "",
        extra: dict[str, Any] | None = None,
    ) -> SourceReviewEvent:
        timestamp = utc_now()
        normalized_mode = mode.value if isinstance(mode, SourceReviewMode) else str(mode)
        normalized_status = (
            review_status.value if isinstance(review_status, SourceStatus) else review_status
        )
        normalized_variant = (
            map_variant.value if isinstance(map_variant, MapVariant) else map_variant
        )
        normalized_tags = normalize_fallacy_tags(fallacy_tags)
        rubric_model = rubric or ReviewRubric()
        payload_extra = extra or {}
        with connect(str(self.settings.db_path)) as conn:
            review_id = self.next_id(conn)
            conn.execute(
                """
                INSERT INTO source_reviews(
                  id, source_id, reviewer, mode, review_status, map_variant, summary, notes,
                  fallacy_tags_json, rubric_json, llm_provider, llm_model, extra_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review_id,
                    source_id,
                    reviewer,
                    normalized_mode,
                    normalized_status,
                    normalized_variant,
                    summary,
                    notes,
                    json.dumps(normalized_tags),
                    rubric_model.model_dump_json(),
                    llm_provider,
                    llm_model,
                    json.dumps(payload_extra),
                    timestamp,
                ),
            )
            conn.commit()
        return self.get_source_review_event(review_id)

    def create_source(
        self,
        *,
        direction_slug: str,
        title: str,
        authors: str = "",
        year: str = "",
        url: str = "",
        source_type: SourceType = SourceType.SELF_PUBLISHED,
        claim_type: SourceClaimType = SourceClaimType.PROOF_ATTEMPT,
        review_status: SourceStatus = SourceStatus.INTAKE,
        map_variant: MapVariant = MapVariant.UNSPECIFIED,
        summary: str = "",
        notes: str = "",
        fallacy_tags: list[str] | None = None,
        rubric: ReviewRubric | None = None,
    ) -> Source:
        timestamp = utc_now()
        tags = normalize_fallacy_tags(fallacy_tags)
        rubric_model = rubric or ReviewRubric()
        with connect(str(self.settings.db_path)) as conn:
            source_id = self.next_id(conn)
            conn.execute(
                """
                INSERT INTO sources(
                  id, direction_slug, title, authors, year, url,
                  source_type, claim_type, review_status, map_variant, summary, notes,
                  fallacy_tags_json, rubric_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_id,
                    direction_slug,
                    title,
                    authors,
                    year,
                    url,
                    source_type.value,
                    claim_type.value,
                    review_status.value,
                    map_variant.value,
                    summary,
                    notes,
                    json.dumps(tags),
                    rubric_model.model_dump_json(),
                    timestamp,
                    timestamp,
                ),
            )
            conn.commit()

        source = self.get_source(source_id)
        note_path = self._write_source_note(source)
        self.create_artifact(
            kind=ArtifactKind.NOTE,
            path=note_path,
            metadata={
                "source_id": source_id,
                "direction": direction_slug,
                "source_type": source_type.value,
                "claim_type": claim_type.value,
            },
        )
        self._record_source_review_event(
            source_id=source.id,
            reviewer="system",
            mode=SourceReviewMode.INTAKE,
            review_status=source.review_status,
            map_variant=source.map_variant,
            summary=source.summary,
            notes=source.notes,
            fallacy_tags=source.fallacy_tags,
            rubric=source.rubric,
        )
        return source

    def update_source_review(
        self,
        source_id: str,
        *,
        review_status: SourceStatus | str | None = None,
        map_variant: MapVariant | str | None = None,
        summary: str | None = None,
        notes: str | None = None,
        fallacy_tags: list[str] | None = None,
        rubric: ReviewRubric | None = None,
    ) -> Source:
        fields: list[str] = ["updated_at = ?"]
        values: list[Any] = [utc_now()]
        if review_status is not None:
            fields.append("review_status = ?")
            values.append(review_status.value if isinstance(review_status, SourceStatus) else review_status)
        if map_variant is not None:
            fields.append("map_variant = ?")
            values.append(map_variant.value if isinstance(map_variant, MapVariant) else map_variant)
        if summary is not None:
            fields.append("summary = ?")
            values.append(summary)
        if notes is not None:
            fields.append("notes = ?")
            values.append(notes)
        if fallacy_tags is not None:
            fields.append("fallacy_tags_json = ?")
            values.append(json.dumps(normalize_fallacy_tags(fallacy_tags)))
        if rubric is not None:
            fields.append("rubric_json = ?")
            values.append(rubric.model_dump_json())
        values.append(source_id)
        with connect(str(self.settings.db_path)) as conn:
            conn.execute(
                f"UPDATE sources SET {', '.join(fields)} WHERE id = ?",
                values,
            )
            conn.commit()
        source = self.get_source(source_id)
        note_path = self._write_source_note(source)
        review_path = self._write_source_review_snapshot(source)
        self.create_artifact(
            kind=ArtifactKind.NOTE,
            path=note_path,
            metadata={
                "source_id": source.id,
                "direction": source.direction_slug,
                "source_type": source.source_type.value,
                "claim_type": source.claim_type.value,
                "kind": "source-note-refresh",
            },
        )
        self.create_artifact(
            kind=ArtifactKind.REPORT,
            path=review_path,
            metadata={
                "source_id": source.id,
                "review_status": source.review_status.value,
            },
        )
        self._record_source_review_event(
            source_id=source.id,
            reviewer="integrator",
            mode=SourceReviewMode.MANUAL,
            review_status=source.review_status,
            map_variant=source.map_variant,
            summary=source.summary,
            notes=source.notes,
            fallacy_tags=source.fallacy_tags,
            rubric=source.rubric,
        )
        return source

    def record_source_llm_draft(
        self,
        source_id: str,
        *,
        provider: str,
        model: str,
        review_status: SourceStatus,
        map_variant: MapVariant,
        summary: str,
        notes: str,
        fallacy_tags: list[str],
        rubric: ReviewRubric,
        candidate_claims: list[str],
        deterministic_checks: list[str],
        warnings: list[str],
        raw_text: str = "",
    ) -> SourceReviewEvent:
        return self._record_source_review_event(
            source_id=source_id,
            reviewer="llm",
            mode=SourceReviewMode.LLM_SUGGESTION,
            review_status=review_status,
            map_variant=map_variant,
            summary=summary,
            notes=notes,
            fallacy_tags=fallacy_tags,
            rubric=rubric,
            llm_provider=provider,
            llm_model=model,
            extra={
                "candidate_claims": candidate_claims,
                "deterministic_checks": deterministic_checks,
                "warnings": warnings,
                "raw_text": raw_text,
            },
        )

    def delete_source(self, source_id: str) -> dict[str, object]:
        source = self.get_source(source_id)
        note_relative = f"research/sources/{source.id}.md".replace("\\", "/")
        review_prefix = f"artifacts/reviews/{source.id}-".replace("\\", "/")
        deleted_artifact_ids: list[str] = []
        deleted_files: list[str] = []

        with connect(str(self.settings.db_path)) as conn:
            artifact_rows = conn.execute("SELECT id, path, metadata_json FROM artifacts").fetchall()
            artifact_ids_to_delete: list[str] = []
            artifact_paths_to_delete: list[str] = []
            for row in artifact_rows:
                metadata = json.loads(row["metadata_json"])
                path = str(row["path"]).replace("\\", "/")
                if (
                    metadata.get("source_id") == source_id
                    or path == note_relative
                    or path.startswith(review_prefix)
                ):
                    artifact_ids_to_delete.append(row["id"])
                    artifact_paths_to_delete.append(path)

            conn.execute("DELETE FROM source_reviews WHERE source_id = ?", (source_id,))
            if artifact_ids_to_delete:
                placeholders = ", ".join("?" for _ in artifact_ids_to_delete)
                conn.execute(
                    f"DELETE FROM artifacts WHERE id IN ({placeholders})",
                    artifact_ids_to_delete,
                )
            conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))
            conn.commit()
            deleted_artifact_ids = artifact_ids_to_delete

        for relative_path in artifact_paths_to_delete:
            candidate = self.settings.workspace_root / Path(relative_path)
            if candidate.exists():
                candidate.unlink()
                deleted_files.append(relative_path)

        note_path = self.settings.research_dir / "sources" / f"{source.id}.md"
        if note_path.exists():
            note_path.unlink()
            relative = str(note_path.relative_to(self.settings.workspace_root))
            if relative not in deleted_files:
                deleted_files.append(relative)

        return {
            "id": source_id,
            "deleted": True,
            "deleted_artifact_ids": deleted_artifact_ids,
            "deleted_files": deleted_files,
        }

    def link_claim_run(self, *, claim_id: str, run_id: str, relation: str) -> ClaimRunLink:
        timestamp = utc_now()
        with connect(str(self.settings.db_path)) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO claim_run_links(claim_id, run_id, relation, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (claim_id, run_id, relation, timestamp),
            )
            conn.commit()
        if relation == "supports":
            self.update_claim_status(claim_id, "supported")
        elif relation == "refutes":
            self.update_claim_status(claim_id, "refuted")
        return ClaimRunLink(
            claim_id=claim_id,
            run_id=run_id,
            relation=relation,
            created_at=timestamp,
        )

    def create_artifact(
        self,
        *,
        kind: ArtifactKind,
        path: Path,
        metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
        claim_id: str | None = None,
    ) -> Artifact:
        metadata = metadata or {}
        timestamp = utc_now()
        checksum = sha256_file(path)
        relative_path = str(path.relative_to(self.settings.workspace_root))
        with connect(str(self.settings.db_path)) as conn:
            artifact_id = self.next_id(conn)
            conn.execute(
                """
                INSERT INTO artifacts(id, kind, path, checksum, metadata_json, run_id, claim_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    kind.value,
                    relative_path,
                    checksum,
                    json.dumps(metadata),
                    run_id,
                    claim_id,
                    timestamp,
                ),
            )
            conn.commit()
        return self.get_artifact(artifact_id)

    def list_directions(self) -> list[Direction]:
        with connect(str(self.settings.db_path)) as conn:
            rows = conn.execute("SELECT * FROM directions ORDER BY title").fetchall()
        return [Direction.model_validate(dict(row)) for row in rows]

    def list_tasks(self) -> list[Task]:
        with connect(str(self.settings.db_path)) as conn:
            rows = conn.execute(
                "SELECT * FROM tasks ORDER BY priority ASC, created_at DESC, id DESC"
            ).fetchall()
        return [Task.model_validate(dict(row)) for row in rows]

    def list_runs(self) -> list[Run]:
        with connect(str(self.settings.db_path)) as conn:
            rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC, id DESC").fetchall()
        return [self._row_to_run(row) for row in rows]

    def list_claims(self, *, exclude_direction: str | None = None) -> list[Claim]:
        with connect(str(self.settings.db_path)) as conn:
            if exclude_direction:
                rows = conn.execute(
                    "SELECT * FROM claims WHERE direction_slug != ? ORDER BY created_at DESC, id DESC",
                    (exclude_direction,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM claims ORDER BY created_at DESC, id DESC").fetchall()
        return [self._row_to_claim(row) for row in rows]

    def list_hypotheses(self) -> list[Claim]:
        """List only hypothesis-sandbox claims (separated from main claims)."""
        with connect(str(self.settings.db_path)) as conn:
            rows = conn.execute(
                "SELECT * FROM claims WHERE direction_slug = 'hypothesis-sandbox' "
                "ORDER BY created_at DESC, id DESC"
            ).fetchall()
        return [self._row_to_claim(row) for row in rows]

    def list_claim_run_links(self) -> list[ClaimRunLink]:
        with connect(str(self.settings.db_path)) as conn:
            rows = conn.execute(
                "SELECT * FROM claim_run_links ORDER BY created_at DESC"
            ).fetchall()
        return [ClaimRunLink.model_validate(dict(row)) for row in rows]

    def list_artifacts(self) -> list[Artifact]:
        with connect(str(self.settings.db_path)) as conn:
            rows = conn.execute(
                "SELECT * FROM artifacts ORDER BY created_at DESC, id DESC"
            ).fetchall()
        return [self._row_to_artifact(row) for row in rows]

    def resolve_artifact_path(self, artifact_id: str) -> tuple[Artifact, Path]:
        artifact = self.get_artifact(artifact_id)
        candidate = self.settings.workspace_root / Path(artifact.path)
        resolved = ensure_within_workspace(self.settings.workspace_root, candidate)
        if not resolved.exists():
            raise FileNotFoundError(f"Artifact file missing: {artifact.path}")
        return artifact, resolved

    def read_artifact_content(self, artifact_id: str) -> dict[str, Any]:
        artifact, resolved = self.resolve_artifact_path(artifact_id)
        try:
            content = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = resolved.read_text(encoding="utf-8", errors="replace")
        return {
            "artifact": artifact.model_dump(mode="json"),
            "filename": resolved.name,
            "text": content,
            "content": content,
        }

    def list_workers(self) -> list[Worker]:
        with connect(str(self.settings.db_path)) as conn:
            rows = conn.execute(
                "SELECT * FROM workers ORDER BY updated_at DESC, created_at DESC"
            ).fetchall()
        return [self._row_to_worker(row) for row in rows]

    def list_sources(self) -> list[Source]:
        with connect(str(self.settings.db_path)) as conn:
            rows = conn.execute(
                "SELECT * FROM sources ORDER BY updated_at DESC, created_at DESC"
            ).fetchall()
        return [self._row_to_source(row) for row in rows]

    def list_source_reviews(self, source_id: str) -> list[SourceReviewEvent]:
        self.get_source(source_id)
        with connect(str(self.settings.db_path)) as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM source_reviews
                WHERE source_id = ?
                ORDER BY created_at DESC, id DESC
                """,
                (source_id,),
            ).fetchall()
        return [self._row_to_source_review(row) for row in rows]

    def get_task(self, task_id: str) -> Task:
        with connect(str(self.settings.db_path)) as conn:
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(f"Task not found: {task_id}")
        return Task.model_validate(dict(row))

    def get_run(self, run_id: str) -> Run:
        with connect(str(self.settings.db_path)) as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"Run not found: {run_id}")
        return self._row_to_run(row)

    def get_claim(self, claim_id: str) -> Claim:
        with connect(str(self.settings.db_path)) as conn:
            row = conn.execute(
                "SELECT * FROM claims WHERE id = ?", (claim_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Claim not found: {claim_id}")
        return self._row_to_claim(row)

    def get_artifact(self, artifact_id: str) -> Artifact:
        with connect(str(self.settings.db_path)) as conn:
            row = conn.execute(
                "SELECT * FROM artifacts WHERE id = ?", (artifact_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Artifact not found: {artifact_id}")
        return self._row_to_artifact(row)

    def get_direction(self, slug: str) -> Direction:
        with connect(str(self.settings.db_path)) as conn:
            row = conn.execute(
                "SELECT * FROM directions WHERE slug = ?", (slug,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Direction not found: {slug}")
        return Direction.model_validate(dict(row))

    def get_worker(self, worker_id: str) -> Worker:
        with connect(str(self.settings.db_path)) as conn:
            row = conn.execute(
                "SELECT * FROM workers WHERE id = ?",
                (worker_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Worker not found: {worker_id}")
        return self._row_to_worker(row)

    def get_worker_by_name(self, name: str) -> Worker:
        with connect(str(self.settings.db_path)) as conn:
            row = conn.execute(
                """
                SELECT *
                FROM workers
                WHERE name = ?
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 1
                """,
                (name,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Worker not found: {name}")
        return self._row_to_worker(row)

    def get_source(self, source_id: str) -> Source:
        with connect(str(self.settings.db_path)) as conn:
            row = conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
        if row is None:
            raise KeyError(f"Source not found: {source_id}")
        return self._row_to_source(row)

    def get_source_review_event(self, review_id: str) -> SourceReviewEvent:
        with connect(str(self.settings.db_path)) as conn:
            row = conn.execute(
                "SELECT * FROM source_reviews WHERE id = ?",
                (review_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Source review not found: {review_id}")
        return self._row_to_source_review(row)

    def claim_next_run(
        self,
        *,
        worker_id: str,
        supported_hardware: list[str],
        supported_kernels: list[str],
    ) -> Run | None:
        if not supported_hardware or not supported_kernels:
            return None

        hardware_placeholders = ", ".join("?" for _ in supported_hardware)
        kernel_placeholders = ", ".join("?" for _ in supported_kernels)
        timestamp = utc_now()
        query = f"""
            SELECT *
            FROM runs
            WHERE status = ?
              AND hardware IN ({hardware_placeholders})
              AND kernel IN ({kernel_placeholders})
            ORDER BY created_at ASC
            LIMIT 1
        """

        with connect(str(self.settings.db_path)) as conn:
            row = conn.execute(
                query,
                [RunStatus.QUEUED.value, *supported_hardware, *supported_kernels],
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    UPDATE workers
                    SET status = ?, current_run_id = ?, updated_at = ?, last_heartbeat_at = ?
                    WHERE id = ?
                    """,
                    (
                        WorkerStatus.IDLE.value,
                        None,
                        timestamp,
                        timestamp,
                        worker_id,
                    ),
                )
                conn.commit()
                return None

            updated = conn.execute(
                """
                UPDATE runs
                SET status = ?, started_at = ?
                WHERE id = ? AND status = ?
                """,
                (
                    RunStatus.RUNNING.value,
                    timestamp,
                    row["id"],
                    RunStatus.QUEUED.value,
                ),
            )
            if updated.rowcount != 1:
                conn.rollback()
                return None

            conn.execute(
                """
                UPDATE workers
                SET status = ?, current_run_id = ?, updated_at = ?, last_heartbeat_at = ?
                WHERE id = ?
                """,
                (
                    WorkerStatus.RUNNING.value,
                    row["id"],
                    timestamp,
                    timestamp,
                    worker_id,
                ),
            )
            conn.commit()

        return self.get_run(row["id"])

    def summary(self) -> DashboardSummary:
        with connect(str(self.settings.db_path)) as conn:
            direction_count = conn.execute(
                "SELECT COUNT(*) AS count FROM directions"
            ).fetchone()["count"]
            run_count = conn.execute(
                "SELECT COUNT(*) AS count FROM runs"
            ).fetchone()["count"]
            validated_run_count = conn.execute(
                "SELECT COUNT(*) AS count FROM runs WHERE status = ?",
                (RunStatus.VALIDATED.value,),
            ).fetchone()["count"]
            queued_run_count = conn.execute(
                "SELECT COUNT(*) AS count FROM runs WHERE status = ?",
                (RunStatus.QUEUED.value,),
            ).fetchone()["count"]
            running_run_count = conn.execute(
                "SELECT COUNT(*) AS count FROM runs WHERE status = ?",
                (RunStatus.RUNNING.value,),
            ).fetchone()["count"]
            claim_count = conn.execute(
                "SELECT COUNT(*) AS count FROM claims"
            ).fetchone()["count"]
            open_task_count = conn.execute(
                "SELECT COUNT(*) AS count FROM tasks WHERE status = ?",
                (TaskStatus.OPEN.value,),
            ).fetchone()["count"]
            artifact_count = conn.execute(
                "SELECT COUNT(*) AS count FROM artifacts"
            ).fetchone()["count"]
            source_count = conn.execute(
                "SELECT COUNT(*) AS count FROM sources"
            ).fetchone()["count"]
            flagged_source_count = conn.execute(
                "SELECT COUNT(*) AS count FROM sources WHERE review_status = ?",
                (SourceStatus.FLAGGED.value,),
            ).fetchone()["count"]
            worker_count = conn.execute(
                "SELECT COUNT(*) AS count FROM workers"
            ).fetchone()["count"]
            active_worker_count = conn.execute(
                "SELECT COUNT(*) AS count FROM workers WHERE status = ?",
                (WorkerStatus.RUNNING.value,),
            ).fetchone()["count"]
            timestamps = []
            for query in (
                "SELECT MAX(updated_at) AS value FROM directions",
                "SELECT MAX(updated_at) AS value FROM tasks",
                "SELECT MAX(finished_at) AS value FROM runs",
                "SELECT MAX(updated_at) AS value FROM claims",
                "SELECT MAX(updated_at) AS value FROM sources",
                "SELECT MAX(created_at) AS value FROM artifacts",
                "SELECT MAX(updated_at) AS value FROM workers",
            ):
                row = conn.execute(query).fetchone()
                if row and row["value"]:
                    timestamps.append(row["value"])
        return DashboardSummary(
            direction_count=direction_count,
            run_count=run_count,
            validated_run_count=validated_run_count,
            queued_run_count=queued_run_count,
            running_run_count=running_run_count,
            claim_count=claim_count,
            open_task_count=open_task_count,
            artifact_count=artifact_count,
            worker_count=worker_count,
            active_worker_count=active_worker_count,
            source_count=source_count,
            flagged_source_count=flagged_source_count,
            latest_write_at=max(timestamps) if timestamps else None,
        )

    def get_compute_profile(self) -> ComputeProfile:
        with connect(str(self.settings.db_path)) as conn:
            row = conn.execute(
                "SELECT value_json, updated_at FROM runtime_settings WHERE key = ?",
                ("compute_profile",),
            ).fetchone()
        payload = {"system_percent": 100, "cpu_percent": 100, "gpu_percent": 100}
        updated_at = None
        if row:
            payload.update(json.loads(row["value_json"] or "{}"))
            updated_at = row["updated_at"]
        payload["updated_at"] = updated_at
        return ComputeProfile.model_validate(payload)

    def update_compute_profile(
        self,
        *,
        continuous_enabled: bool = True,
        system_percent: int,
        cpu_percent: int,
        gpu_percent: int,
    ) -> ComputeProfile:
        timestamp = utc_now()
        payload = {
            "continuous_enabled": bool(continuous_enabled),
            "system_percent": max(0, min(100, int(system_percent))),
            "cpu_percent": max(0, min(100, int(cpu_percent))),
            "gpu_percent": max(0, min(100, int(gpu_percent))),
        }
        with connect(str(self.settings.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO runtime_settings(key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json, updated_at = excluded.updated_at
                """,
                ("compute_profile", json.dumps(payload), timestamp),
            )
            conn.commit()
        return self.get_compute_profile()

    def consensus_baseline(self) -> ConsensusBaseline:
        return CONSENSUS_BASELINE

    def review_direction(self, slug: str) -> DirectionReview:
        self.get_direction(slug)
        with connect(str(self.settings.db_path)) as conn:
            validated_runs = int(
                conn.execute(
                    "SELECT COUNT(*) AS count FROM runs WHERE direction_slug = ? AND status = ?",
                    (slug, RunStatus.VALIDATED.value),
                ).fetchone()["count"]
            )
            failed_runs = int(
                conn.execute(
                    "SELECT COUNT(*) AS count FROM runs WHERE direction_slug = ? AND status = ?",
                    (slug, RunStatus.FAILED.value),
                ).fetchone()["count"]
            )
            linked_runs = int(
                conn.execute(
                    """
                    SELECT COUNT(DISTINCT crl.run_id) AS count
                    FROM claim_run_links crl
                    JOIN claims c ON c.id = crl.claim_id
                    WHERE c.direction_slug = ?
                    """,
                    (slug,),
                ).fetchone()["count"]
            )
            promising_claims = int(
                conn.execute(
                    "SELECT COUNT(*) AS count FROM claims WHERE direction_slug = ? AND status = ?",
                    (slug, "promising"),
                ).fetchone()["count"]
            )
            supported_claims = int(
                conn.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM claims
                    WHERE direction_slug = ? AND status IN (?, ?)
                    """,
                    (slug, "supported", "formalize"),
                ).fetchone()["count"]
            )
            refuted_claims = int(
                conn.execute(
                    "SELECT COUNT(*) AS count FROM claims WHERE direction_slug = ? AND status = ?",
                    (slug, "refuted"),
                ).fetchone()["count"]
            )

        if refuted_claims > 0:
            status = DirectionStatus.REFUTED
            rationale = "Refuted claims exist for this direction."
        elif validated_runs >= 2 and (promising_claims + supported_claims) > 0:
            status = DirectionStatus.PROMISING
            rationale = "Multiple validated runs and supported claims exist."
        elif failed_runs >= 2 and validated_runs == 0 and linked_runs == 0:
            status = DirectionStatus.FROZEN
            rationale = "Repeated failed runs without supported evidence."
        else:
            status = DirectionStatus.ACTIVE
            rationale = "Direction remains active pending more evidence."

        score = (validated_runs * 2) + (promising_claims * 2) + (supported_claims * 3) - (
            refuted_claims * 4
        ) - failed_runs
        updated = utc_now()

        with connect(str(self.settings.db_path)) as conn:
            conn.execute(
                "UPDATE directions SET status = ?, score = ?, updated_at = ? WHERE slug = ?",
                (status.value, float(score), updated, slug),
            )
            conn.commit()

        direction = self.get_direction(slug)
        return DirectionReview(
            direction=direction,
            validated_runs=validated_runs,
            promising_claims=promising_claims,
            supported_claims=supported_claims,
            refuted_claims=refuted_claims,
            failed_runs=failed_runs,
            linked_runs=linked_runs,
            rationale=rationale,
        )

    def _row_to_run(self, row) -> Run:
        payload = dict(row)
        payload["checkpoint"] = json.loads(payload.pop("checkpoint_json"))
        payload["metrics"] = json.loads(payload.pop("metrics_json"))
        return Run.model_validate(payload)

    def _row_to_claim(self, row) -> Claim:
        payload = dict(row)
        payload["dependencies"] = json.loads(payload.pop("dependencies_json"))
        return Claim.model_validate(payload)

    def _row_to_artifact(self, row) -> Artifact:
        payload = dict(row)
        payload["metadata"] = json.loads(payload.pop("metadata_json"))
        return Artifact.model_validate(payload)

    def _row_to_worker(self, row) -> Worker:
        payload = dict(row)
        payload["capabilities"] = [
            HardwareCapability.model_validate(item)
            for item in json.loads(payload.pop("capabilities_json"))
        ]
        return Worker.model_validate(payload)

    def _row_to_source(self, row) -> Source:
        payload = dict(row)
        payload["fallacy_tags"] = json.loads(payload.pop("fallacy_tags_json"))
        payload["rubric"] = ReviewRubric.model_validate(json.loads(payload.pop("rubric_json")))
        return Source.model_validate(payload)

    def _row_to_source_review(self, row) -> SourceReviewEvent:
        payload = dict(row)
        payload["fallacy_tags"] = json.loads(payload.pop("fallacy_tags_json"))
        payload["rubric"] = ReviewRubric.model_validate(json.loads(payload.pop("rubric_json")))
        payload["extra"] = json.loads(payload.pop("extra_json"))
        return SourceReviewEvent.model_validate(payload)
