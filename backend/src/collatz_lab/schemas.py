from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class DirectionStatus(StrEnum):
    ACTIVE = "active"
    PROMISING = "promising"
    FROZEN = "frozen"
    REFUTED = "refuted"


class ClaimStatus(StrEnum):
    IDEA = "idea"
    ACTIVE = "active"
    PROMISING = "promising"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    FROZEN = "frozen"
    FORMALIZE = "formalize"


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    VALIDATED = "validated"
    FAILED = "failed"


class TaskStatus(StrEnum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FROZEN = "frozen"


class WorkerStatus(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    OFFLINE = "offline"


class ArtifactKind(StrEnum):
    JSON = "json"
    REPORT = "report"
    NOTE = "note"


class SourceType(StrEnum):
    PEER_REVIEWED = "peer_reviewed"
    PREPRINT = "preprint"
    SELF_PUBLISHED = "self_published"
    BLOG = "blog"
    FORUM = "forum"
    QA = "qa"
    WIKI = "wiki"
    MEDIA = "media"
    INTERNAL = "internal"


class SourceStatus(StrEnum):
    INTAKE = "intake"
    UNDER_REVIEW = "under_review"
    FLAGGED = "flagged"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    CONTEXT = "context"


class SourceClaimType(StrEnum):
    OPEN_PROBLEM_CONSENSUS = "open_problem_consensus"
    PARTIAL_RESULT = "partial_result"
    COMPUTATIONAL_VERIFICATION = "computational_verification"
    PROOF_ATTEMPT = "proof_attempt"
    HEURISTIC = "heuristic"
    DISCUSSION = "discussion"


class MapVariant(StrEnum):
    UNSPECIFIED = "unspecified"
    STANDARD = "standard"
    SHORTCUT = "shortcut"
    ODD_ONLY = "odd_only"
    INVERSE_TREE = "inverse_tree"


class ReviewRubric(BaseModel):
    peer_reviewed: bool | None = None
    acknowledged_errors: bool | None = None
    defines_map_variant: bool | None = None
    distinguishes_empirical_from_proof: bool | None = None
    proves_descent: bool | None = None
    proves_cycle_exclusion: bool | None = None
    uses_statistical_argument: bool | None = None
    validation_backed: bool | None = None
    notes: str = ""


class FallacyTagInfo(BaseModel):
    tag: str
    label: str
    description: str


class Task(BaseModel):
    id: str
    direction_slug: str
    title: str
    kind: str
    description: str
    owner: str
    status: TaskStatus
    priority: int
    created_at: str
    updated_at: str


class Direction(BaseModel):
    id: str
    slug: str
    title: str
    description: str
    owner: str
    status: DirectionStatus
    score: float
    success_criteria: str
    abandon_criteria: str
    created_at: str
    updated_at: str


class Run(BaseModel):
    id: str
    direction_slug: str
    name: str
    status: RunStatus
    range_start: int
    range_end: int
    kernel: str
    owner: str
    checkpoint: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    summary: str
    code_version: str
    hardware: str
    checksum: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None


class Claim(BaseModel):
    id: str
    direction_slug: str
    title: str
    statement: str
    status: ClaimStatus
    owner: str
    dependencies: list[str] = Field(default_factory=list)
    notes: str = ""
    created_at: str
    updated_at: str


class Source(BaseModel):
    id: str
    direction_slug: str
    title: str
    authors: str = ""
    year: str = ""
    url: str = ""
    source_type: SourceType
    claim_type: SourceClaimType
    review_status: SourceStatus
    map_variant: MapVariant = MapVariant.UNSPECIFIED
    summary: str = ""
    notes: str = ""
    fallacy_tags: list[str] = Field(default_factory=list)
    rubric: ReviewRubric = Field(default_factory=ReviewRubric)
    created_at: str
    updated_at: str


class Artifact(BaseModel):
    id: str
    kind: ArtifactKind
    path: str
    checksum: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None
    claim_id: str | None = None
    created_at: str


class ClaimRunLink(BaseModel):
    claim_id: str
    run_id: str
    relation: str
    created_at: str


class HardwareCapability(BaseModel):
    kind: str
    slug: str
    label: str
    available: bool
    supported_hardware: list[str] = Field(default_factory=list)
    supported_kernels: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Worker(BaseModel):
    id: str
    name: str
    role: str
    status: WorkerStatus
    hardware: str
    capabilities: list[HardwareCapability] = Field(default_factory=list)
    current_run_id: str | None = None
    created_at: str
    updated_at: str
    last_heartbeat_at: str | None = None


class ConsensusBaselineItem(BaseModel):
    title: str
    detail: str
    source_url: str


class ConsensusBaseline(BaseModel):
    problem_status: str
    checked_as_of: str
    verified_up_to: str
    note: str
    items: list[ConsensusBaselineItem] = Field(default_factory=list)


class RedditFeedPost(BaseModel):
    id: str
    title: str
    author: str
    permalink: str
    created_at: str
    score: int = 0
    num_comments: int = 0
    flair_text: str = ""
    signal: str = "watch"
    excerpt: str = ""


class RedditFeed(BaseModel):
    subreddit: str
    sort: str
    fetched_at: str
    review_candidate_count: int = 0
    posts: list[RedditFeedPost] = Field(default_factory=list)


class DashboardSummary(BaseModel):
    direction_count: int
    run_count: int
    validated_run_count: int
    queued_run_count: int = 0
    running_run_count: int = 0
    claim_count: int
    open_task_count: int
    artifact_count: int
    worker_count: int = 0
    active_worker_count: int = 0
    source_count: int = 0
    flagged_source_count: int = 0
    latest_write_at: str | None = None


class DirectionReview(BaseModel):
    direction: Direction
    validated_runs: int
    promising_claims: int
    supported_claims: int
    refuted_claims: int
    failed_runs: int
    linked_runs: int
    rationale: str


class ModularProbeResult(BaseModel):
    modulus: int
    allowed_residues: list[int] = Field(default_factory=list)
    checked_limit: int
    checked_odd_values: int
    first_counterexample: int | None = None
    counterexamples: list[int] = Field(default_factory=list)
    rationale: str
