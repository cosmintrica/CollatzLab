from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import Settings
from .hardware import discover_hardware, validate_execution_request
from .llm import get_llm_provider_status
from .reddit_feed import fetch_subreddit_feed
from .repository import LabRepository
from .schemas import (
    Artifact,
    Claim,
    ClaimRunLink,
    ConsensusBaseline,
    DirectionReview,
    FallacyTagInfo,
    MapVariant,
    ModularProbeResult,
    RedditFeed,
    ReviewRubric,
    Run,
    Source,
    SourceClaimType,
    SourceStatus,
    SourceType,
    Task,
    Worker,
)
from .services import execute_run, generate_report, probe_modular_claim, validate_run


class CreateTaskRequest(BaseModel):
    direction_slug: str
    title: str
    kind: str
    description: str
    owner: str = "integrator"
    priority: int = 2


class CreateRunRequest(BaseModel):
    direction_slug: str
    name: str
    range_start: int
    range_end: int
    kernel: str = "cpu-direct"
    owner: str = "compute-agent"
    code_version: str = "workspace"
    hardware: str = "cpu"
    enqueue_only: bool = False


class CreateClaimRequest(BaseModel):
    direction_slug: str
    title: str
    statement: str
    owner: str = "theory-agent"
    dependencies: list[str] = []
    notes: str = ""


class LinkClaimRunRequest(BaseModel):
    claim_id: str
    run_id: str
    relation: str


class CreateSourceRequest(BaseModel):
    direction_slug: str
    title: str
    authors: str = ""
    year: str = ""
    url: str = ""
    source_type: SourceType = SourceType.SELF_PUBLISHED
    claim_type: SourceClaimType = SourceClaimType.PROOF_ATTEMPT
    review_status: SourceStatus = SourceStatus.INTAKE
    map_variant: MapVariant = MapVariant.UNSPECIFIED
    summary: str = ""
    notes: str = ""
    fallacy_tags: list[str] = []
    rubric: ReviewRubric = ReviewRubric()


class ReviewSourceRequest(BaseModel):
    review_status: SourceStatus | None = None
    map_variant: MapVariant | None = None
    summary: str | None = None
    notes: str | None = None
    fallacy_tags: list[str] | None = None
    rubric: ReviewRubric | None = None


class ModularProbeRequest(BaseModel):
    modulus: int
    allowed_residues: list[int]
    search_limit: int = 2048


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    repository = LabRepository(settings)
    repository.init()

    app = FastAPI(title="Collatz Lab API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.repository = repository

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/summary")
    def summary():
        return repository.summary()

    @app.get("/api/directions")
    def list_directions():
        return repository.list_directions()

    @app.get("/api/hardware")
    def hardware():
        return discover_hardware()

    @app.get("/api/llm/status")
    def llm_status():
        return get_llm_provider_status()

    @app.post("/api/directions/{slug}/review", response_model=DirectionReview)
    def review_direction(slug: str):
        try:
            return repository.review_direction(slug)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/tasks")
    def list_tasks():
        return repository.list_tasks()

    @app.post("/api/tasks", response_model=Task)
    def create_task(request: CreateTaskRequest):
        return repository.create_task(
            direction_slug=request.direction_slug,
            title=request.title,
            kind=request.kind,
            description=request.description,
            owner=request.owner,
            priority=request.priority,
        )

    @app.get("/api/runs")
    def list_runs():
        return repository.list_runs()

    @app.post("/api/runs", response_model=Run)
    def create_run(request: CreateRunRequest):
        try:
            validate_execution_request(
                requested_hardware=request.hardware,
                requested_kernel=request.kernel,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        run = repository.create_run(
            direction_slug=request.direction_slug,
            name=request.name,
            range_start=request.range_start,
            range_end=request.range_end,
            kernel=request.kernel,
            owner=request.owner,
            code_version=request.code_version,
            hardware=request.hardware,
        )
        if request.enqueue_only:
            return run
        return execute_run(repository, run.id)

    @app.post("/api/runs/{run_id}/resume", response_model=Run)
    def resume_run(run_id: str):
        try:
            return execute_run(repository, run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/runs/{run_id}/validate", response_model=Run)
    def validate(run_id: str):
        try:
            return validate_run(repository, run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/claims")
    def list_claims():
        return repository.list_claims()

    @app.get("/api/claim-run-links", response_model=list[ClaimRunLink])
    def list_claim_run_links():
        return repository.list_claim_run_links()

    @app.post("/api/claims", response_model=Claim)
    def create_claim(request: CreateClaimRequest):
        return repository.create_claim(
            direction_slug=request.direction_slug,
            title=request.title,
            statement=request.statement,
            owner=request.owner,
            dependencies=request.dependencies,
            notes=request.notes,
        )

    @app.post("/api/claims/link-run")
    def link_claim_run(request: LinkClaimRunRequest):
        return repository.link_claim_run(
            claim_id=request.claim_id,
            run_id=request.run_id,
            relation=request.relation,
        )

    @app.get("/api/sources", response_model=list[Source])
    def list_sources():
        return repository.list_sources()

    @app.post("/api/sources", response_model=Source)
    def create_source(request: CreateSourceRequest):
        try:
            return repository.create_source(
                direction_slug=request.direction_slug,
                title=request.title,
                authors=request.authors,
                year=request.year,
                url=request.url,
                source_type=request.source_type,
                claim_type=request.claim_type,
                review_status=request.review_status,
                map_variant=request.map_variant,
                summary=request.summary,
                notes=request.notes,
                fallacy_tags=request.fallacy_tags,
                rubric=request.rubric,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/sources/{source_id}/review", response_model=Source)
    def review_source(source_id: str, request: ReviewSourceRequest):
        try:
            return repository.update_source_review(
                source_id,
                review_status=request.review_status,
                map_variant=request.map_variant,
                summary=request.summary,
                notes=request.notes,
                fallacy_tags=request.fallacy_tags,
                rubric=request.rubric,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/review/fallacy-tags", response_model=list[FallacyTagInfo])
    def fallacy_tags():
        return repository.list_fallacy_tags()

    @app.get("/api/consensus-baseline", response_model=ConsensusBaseline)
    def consensus_baseline():
        return repository.consensus_baseline()

    @app.get("/api/external/reddit/collatz", response_model=RedditFeed)
    def reddit_collatz_feed(limit: int = 8, sort: str = "new"):
        try:
            return fetch_subreddit_feed(subreddit="Collatz", sort=sort, limit=limit)
        except Exception as exc:  # pragma: no cover - runtime network fallback
            raise HTTPException(status_code=502, detail=f"Unable to fetch Reddit feed: {exc}") from exc

    @app.post("/api/review/probes/modular", response_model=ModularProbeResult)
    def modular_probe(request: ModularProbeRequest):
        try:
            return probe_modular_claim(
                modulus=request.modulus,
                allowed_residues=request.allowed_residues,
                search_limit=request.search_limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/artifacts")
    def list_artifacts():
        return repository.list_artifacts()

    @app.get("/api/artifacts/{artifact_id}/content")
    def artifact_content(artifact_id: str):
        try:
            return repository.read_artifact_content(artifact_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/artifacts/{artifact_id}/download")
    def artifact_download(artifact_id: str):
        try:
            artifact, path = repository.resolve_artifact_path(artifact_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path, filename=path.name, media_type="application/octet-stream")

    @app.get("/api/workers", response_model=list[Worker])
    def list_workers():
        return repository.list_workers()

    @app.get("/api/workers/capabilities")
    def worker_capabilities():
        return discover_hardware()

    @app.post("/api/reports/generate")
    def report():
        path = generate_report(repository)
        return {"path": str(path.relative_to(repository.settings.workspace_root))}

    return app
