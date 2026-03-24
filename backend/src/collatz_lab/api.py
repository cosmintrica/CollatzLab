from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .autopilot import AutopilotManager, run_autopilot_cycle
from .config import Settings, write_env_updates
from .hardware import discover_hardware, validate_execution_request
from .llm import (
    generate_source_review_draft,
    get_llm_provider_status,
)
from .reddit_feed import fetch_subreddit_feed
from .repository import LabRepository
from .hypothesis import (
    analyze_glide_structure,
    analyze_record_seeds,
    analyze_residue_classes,
    analyze_residue_classes_stratified,
    run_battery_scalability_report,
    run_hypothesis_battery,
    scan_trajectory_depths,
    test_stopping_time_growth,
)
from .schemas import (
    Artifact,
    Claim,
    ClaimRunLink,
    ComputeProfile,
    ConsensusBaseline,
    DirectionReview,
    FallacyTagInfo,
    Hypothesis,
    LLMAutopilotRun,
    LLMAutopilotStatus,
    LLMReviewDraft,
    MapVariant,
    ModularProbeResult,
    RedditFeed,
    ReviewRubric,
    Run,
    Source,
    SourceClaimType,
    SourceReviewEvent,
    SourceStatus,
    SourceType,
    Task,
    Worker,
)
from .logutil import is_noise_log_entry
from .services import execute_run, generate_report, probe_modular_claim
from .validation import validate_run


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


class LLMSetupRequest(BaseModel):
    api_key: str
    enabled: bool = True
    model: str = "gemini-2.5-flash"


class LLMAutopilotRequest(BaseModel):
    apply: bool = False
    max_tasks: int = 3


class LLMAutopilotConfigRequest(BaseModel):
    enabled: bool
    interval_seconds: int = 7200
    max_tasks: int = 2


class ComputeProfileRequest(BaseModel):
    continuous_enabled: bool = True
    system_percent: int = 100
    cpu_percent: int = 100
    gpu_percent: int = 100


class HypothesisBatteryRequest(BaseModel):
    end: int = 50_000
    # Omit modulus 8 here: the battery always runs stratified mod-8 separately (richer bins).
    moduli: list[int] = Field(default_factory=lambda: [3, 4, 6, 12, 16, 24])


class HypothesisBatteryStabilityRequest(BaseModel):
    """Run the same structural probes at multiple range ends; flag status changes."""

    endpoints: list[int] = Field(default_factory=lambda: [50_000, 200_000, 1_000_000])
    glide_sample_cap: int = 8_000
    stratified_bin_count: int = 8
    persist: bool = False


class ResidueClassRequest(BaseModel):
    modulus: int
    start: int = 1
    end: int = 100_000
    odd_only: bool = True


class RecordSeedRequest(BaseModel):
    start: int = 1
    end: int = 100_000


class TrajectoryDepthRequest(BaseModel):
    start: int = 1
    end: int = 50_000
    top_k: int = 20


class GrowthRateRequest(BaseModel):
    start: int = 1
    end: int = 100_000
    bin_count: int = 20


class StratifiedResidueRequest(BaseModel):
    modulus: int = 8
    start: int = 1
    end: int = 100_000
    odd_only: bool = True
    bin_count: int = 8
    z_threshold: float = 2.0
    min_bins_consistent: int = 3
    odd_stride: int = 1


class GlideStructureRequest(BaseModel):
    start: int = 1
    end: int = 50_000
    modulus: int = 8
    sample_cap: int = 6_000
    deviation_threshold: float = 0.02
    bootstrap_reps: int = 400


class MetalBenchRunRequest(BaseModel):
    """Start a background Metal chunk sweep (macOS + native helper only)."""

    preset: str = "standard"
    quick: bool = True
    linear_end: int = 0
    reps: int = 3
    warmup: int = 1
    chunks_csv: str = ""
    write_calibration: bool = True
    pipeline_ab: bool = False


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    repository = LabRepository(settings)
    repository.init()
    autopilot_manager = AutopilotManager(
        repository,
        enabled=settings.llm_autopilot_enabled,
        interval_seconds=settings.llm_autopilot_interval_seconds,
        max_tasks=settings.llm_autopilot_max_tasks,
    )

    app = FastAPI(title="Collatz Lab API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.repository = repository
    app.state.autopilot_manager = autopilot_manager

    if settings.llm_autopilot_enabled:
        autopilot_manager.start()

    @app.on_event("shutdown")
    def shutdown_background_services() -> None:
        autopilot_manager.stop()

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

    @app.get("/api/compute/profile", response_model=ComputeProfile)
    def compute_profile():
        return repository.get_compute_profile()

    @app.post("/api/compute/profile", response_model=ComputeProfile)
    def update_compute_profile(request: ComputeProfileRequest):
        return repository.update_compute_profile(
            continuous_enabled=request.continuous_enabled,
            system_percent=request.system_percent,
            cpu_percent=request.cpu_percent,
            gpu_percent=request.gpu_percent,
        )

    @app.get("/api/llm/status")
    def llm_status():
        return get_llm_provider_status()

    @app.post("/api/llm/setup")
    def llm_setup(request: LLMSetupRequest):
        api_key = request.api_key.strip()
        if not api_key:
            raise HTTPException(status_code=400, detail="Gemini API key cannot be empty.")
        write_env_updates(
            settings.workspace_root,
            {
                "COLLATZ_LLM_ENABLED": "1" if request.enabled else "0",
                "GEMINI_MODEL": request.model.strip() or "gemini-2.5-flash",
                "GEMINI_API_KEY": api_key,
            },
        )
        if autopilot_manager.status().enabled:
            autopilot_manager.start()
        return get_llm_provider_status()

    @app.get("/api/llm/autopilot/status", response_model=LLMAutopilotStatus)
    def llm_autopilot_status():
        return autopilot_manager.status()

    @app.post("/api/llm/autopilot/config", response_model=LLMAutopilotStatus)
    def llm_autopilot_config(request: LLMAutopilotConfigRequest):
        write_env_updates(
            settings.workspace_root,
            {
                "COLLATZ_LLM_AUTOPILOT_ENABLED": "1" if request.enabled else "0",
                "COLLATZ_LLM_AUTOPILOT_INTERVAL_SECONDS": str(max(60, request.interval_seconds)),
                "COLLATZ_LLM_AUTOPILOT_MAX_TASKS": str(max(1, min(5, request.max_tasks))),
            },
        )
        return autopilot_manager.configure(
            enabled=request.enabled,
            interval_seconds=request.interval_seconds,
            max_tasks=request.max_tasks,
        )

    @app.post("/api/llm/autopilot/run", response_model=LLMAutopilotRun)
    def llm_autopilot_run(request: LLMAutopilotRequest):
        try:
            plan = run_autopilot_cycle(
                repository,
                max_tasks=max(1, min(5, request.max_tasks)),
                apply=request.apply,
                mode="manual",
            )
        except RuntimeError as exc:
            autopilot_manager.record_error(str(exc))
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        autopilot_manager.record_result(plan)
        return plan

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
        # Exclude hypothesis-sandbox claims from main claims list
        return repository.list_claims(exclude_direction="hypothesis-sandbox")

    @app.get("/api/hypotheses", response_model=list[Claim])
    def list_hypotheses():
        """Separate endpoint for hypothesis-sandbox claims."""
        return repository.list_hypotheses()

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

    @app.delete("/api/sources/{source_id}")
    def delete_source(source_id: str):
        try:
            return repository.delete_source(source_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/sources/{source_id}/reviews", response_model=list[SourceReviewEvent])
    def source_review_history(source_id: str):
        try:
            return repository.list_source_reviews(source_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/sources/{source_id}/review-draft", response_model=LLMReviewDraft)
    def source_review_draft(source_id: str):
        try:
            source = repository.get_source(source_id)
            draft = generate_source_review_draft(
                source=source,
                baseline=repository.consensus_baseline(),
                allowed_tags=[item.tag for item in repository.list_fallacy_tags()],
            )
            repository.record_source_llm_draft(
                source_id=source_id,
                provider=draft.provider,
                model=draft.model,
                review_status=draft.review_status,
                map_variant=draft.map_variant,
                summary=draft.summary,
                notes=draft.notes,
                fallacy_tags=draft.fallacy_tags,
                rubric=draft.rubric,
                candidate_claims=draft.candidate_claims,
                deterministic_checks=draft.deterministic_checks,
                warnings=draft.warnings,
                raw_text=draft.raw_text,
            )
            return draft
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

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

    @app.get("/api/workers/gpu-sieve-metal")
    def worker_gpu_sieve_metal_status():
        """Diagnostics for native Metal ``gpu-sieve`` helper (macOS); inert metadata elsewhere."""
        from collatz_lab.gpu_sieve_metal_runtime import diagnostics_native_metal_sieve

        return diagnostics_native_metal_sieve()

    @app.post("/api/workers/gpu-sieve-metal/stdio-shutdown")
    def worker_gpu_sieve_metal_stdio_shutdown():
        """Terminate the long-lived ``metal_sieve_chunk --stdio`` child (frees peak Metal/RAM)."""
        from collatz_lab.gpu_sieve_metal_runtime import shutdown_metal_stdio_transport

        shutdown_metal_stdio_transport()
        return {"ok": True}

    @app.get("/api/workers/cpu-sieve-native")
    def worker_cpu_sieve_native_status():
        """Diagnostics for optional native C ``cpu-sieve`` backend (Darwin/Linux .dylib/.so)."""
        from collatz_lab.cpu_sieve_native_runtime import diagnostics_native_cpu_sieve

        return diagnostics_native_cpu_sieve()

    @app.get("/api/workers/native-stack")
    def worker_native_stack_status():
        """Single JSON: native ``cpu-sieve`` (.dylib/.so) + Metal ``gpu-sieve`` helper (any OS; fields may be inert)."""
        from collatz_lab.cpu_sieve_native_runtime import diagnostics_native_cpu_sieve
        from collatz_lab.gpu_sieve_metal_runtime import diagnostics_native_metal_sieve

        return {
            "cpu_sieve_native": diagnostics_native_cpu_sieve(),
            "gpu_sieve_metal": diagnostics_native_metal_sieve(),
        }

    @app.get("/api/bench/metal-chunk/status")
    def bench_metal_chunk_status():
        """Whether a sweep is running; macOS + Metal helper flags."""
        from collatz_lab.bench_metal_chunk import metal_benchmark_public_status

        return metal_benchmark_public_status()

    @app.get("/api/bench/metal-chunk/history")
    def bench_metal_chunk_history(limit: int = 25):
        """Recent Metal chunk benchmarks saved in local SQLite."""
        return repository.list_metal_benchmark_runs(limit=limit)

    @app.get("/api/bench/metal-chunk/presets")
    def bench_metal_chunk_presets():
        """Fixed, reproducible sweep definitions (same inputs on every machine)."""
        from collatz_lab.bench_metal_presets import list_metal_bench_presets_public

        return list_metal_bench_presets_public()

    @app.get("/api/bench/metal-chunk/hall-of-fame")
    def bench_metal_chunk_hall_of_fame(platform: str = "Darwin", limit: int = 25):
        """Top local Metal chunk throughputs for a given OS (from saved completed runs)."""
        allowed = {"Darwin", "Linux", "Windows"}
        if platform not in allowed:
            raise HTTPException(
                status_code=400,
                detail="platform must be one of: Darwin, Linux, Windows",
            )
        return repository.list_metal_benchmark_hall_of_fame(platform=platform, limit=limit)

    @app.get("/api/bench/metal-chunk/runs/{job_id}")
    def bench_metal_chunk_run_detail(job_id: str):
        row = repository.get_metal_benchmark_run(job_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Benchmark run not found.")
        return row

    @app.post("/api/bench/metal-chunk/run")
    def bench_metal_chunk_run(request: MetalBenchRunRequest):
        """Queue a background benchmark (one at a time per API process)."""
        from collatz_lab.bench_metal_chunk import metal_benchmark_try_start
        from collatz_lab.bench_metal_presets import resolve_metal_bench_params

        raw = request.model_dump()
        body = resolve_metal_bench_params(
            preset=raw["preset"],
            quick=raw["quick"],
            linear_end=raw["linear_end"],
            reps=raw["reps"],
            warmup=raw["warmup"],
            chunks_csv=raw["chunks_csv"],
            write_calibration=raw["write_calibration"],
            pipeline_ab=raw["pipeline_ab"],
        )
        out = metal_benchmark_try_start(repository, body)
        if not out.get("started"):
            msg = out.get("message") or "Could not start benchmark."
            code = out.get("error_code")
            if code == "wrong_platform":
                raise HTTPException(status_code=400, detail=msg)
            raise HTTPException(status_code=409, detail=msg)
        return out

    @app.get("/api/validation/contract")
    def validation_contract():
        """Platform-wide source-of-truth / validation contract metadata (no compute)."""
        from collatz_lab.validation_source import validation_contract_metadata

        return validation_contract_metadata()

    @app.post("/api/reports/generate")
    def report():
        path = generate_report(repository)
        return {"path": str(path.relative_to(repository.settings.workspace_root))}

    # ── Hypothesis Sandbox ──

    @app.post("/api/hypotheses/battery", response_model=list[Hypothesis])
    def hypothesis_battery(request: HypothesisBatteryRequest):
        try:
            return run_hypothesis_battery(
                repository,
                end=request.end,
                moduli=request.moduli,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/hypotheses/battery/stability")
    def hypothesis_battery_stability(request: HypothesisBatteryStabilityRequest):
        try:
            return run_battery_scalability_report(
                endpoints=request.endpoints,
                glide_sample_cap=request.glide_sample_cap,
                stratified_bin_count=request.stratified_bin_count,
                repository=repository if request.persist else None,
                persist=request.persist,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/hypotheses/residue-class")
    def hypothesis_residue_class(request: ResidueClassRequest):
        try:
            result = analyze_residue_classes(
                request.modulus,
                start=request.start,
                end=request.end,
                odd_only=request.odd_only,
            )
            return result.__dict__
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/hypotheses/record-seeds")
    def hypothesis_record_seeds(request: RecordSeedRequest):
        result = analyze_record_seeds(start=request.start, end=request.end)
        return result.__dict__

    @app.post("/api/hypotheses/trajectory-depths")
    def hypothesis_trajectory_depths(request: TrajectoryDepthRequest):
        result = scan_trajectory_depths(
            start=request.start,
            end=request.end,
            top_k=request.top_k,
        )
        return result.__dict__

    @app.post("/api/hypotheses/growth-rate")
    def hypothesis_growth_rate(request: GrowthRateRequest):
        result = test_stopping_time_growth(
            start=request.start,
            end=request.end,
            bin_count=request.bin_count,
        )
        return result.__dict__

    @app.post("/api/hypotheses/stratified-residue")
    def hypothesis_stratified_residue(request: StratifiedResidueRequest):
        try:
            result = analyze_residue_classes_stratified(
                request.modulus,
                start=request.start,
                end=request.end,
                odd_only=request.odd_only,
                bin_count=request.bin_count,
                z_threshold=request.z_threshold,
                min_bins_consistent=request.min_bins_consistent,
                odd_stride=request.odd_stride,
            )
            return result.__dict__
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/hypotheses/glide-structure")
    def hypothesis_glide_structure(request: GlideStructureRequest):
        try:
            result = analyze_glide_structure(
                request.start,
                request.end,
                modulus=request.modulus,
                sample_cap=request.sample_cap,
                deviation_threshold=request.deviation_threshold,
                bootstrap_reps=request.bootstrap_reps,
            )
            return result.__dict__
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/paper")
    def get_paper():
        import json
        from pathlib import Path
        paper_path = Path(__file__).parent.parent.parent.parent / "research" / "paper.json"
        if not paper_path.exists():
            raise HTTPException(status_code=404, detail="paper.json not found")
        return json.loads(paper_path.read_text(encoding="utf-8"))

    @app.get("/api/logs")
    def get_logs(
        q: str = "",
        level: str = "",
        source: str = "",
        limit: int = 500,
    ):
        """Aggregate logs from worker log files and recent failed runs."""
        import re
        from pathlib import Path

        entries: list[dict] = []
        log_dir = settings.workspace_root / "data" / "logs"
        log_line_re = re.compile(
            r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)\s+"
            r"(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s+"
            r"(\S+):\s+(.*)$"
        )

        if log_dir.is_dir():
            for log_file in sorted(log_dir.glob("worker-*.log")):
                worker_name = log_file.stem.replace("worker-", "")
                try:
                    lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
                except OSError:
                    continue
                current: dict | None = None
                for line in lines:
                    m = log_line_re.match(line)
                    if m:
                        if current:
                            entries.append(current)
                        current = {
                            "ts": m.group(1).replace(",", "."),
                            "level": m.group(2),
                            "logger": m.group(3),
                            "msg": m.group(4),
                            "source": worker_name,
                            "kind": "worker",
                        }
                    elif current:
                        current["msg"] += "\n" + line
                if current:
                    entries.append(current)

        # Add recent failed runs as log entries
        for run in repository.list_runs():
            if run.status.value != "failed":
                continue
            summary = run.summary or "No summary"
            # Superseded validation failures are resolved — demote to INFO
            superseded = "Superseded by" in summary and "(completed)" in summary
            ts = run.finished_at or run.created_at or ""
            entries.append({
                "ts": ts.replace("T", " ").replace("+00:00", ""),
                "level": "INFO" if superseded else "ERROR",
                "logger": "run",
                "msg": f"[{run.id}] {summary}",
                "source": run.hardware or "unknown",
                "kind": "run-superseded" if superseded else "run-failure",
            })

        # Filter
        q_lower = q.lower()
        level_upper = level.upper()
        source_lower = source.lower()

        def matches(e: dict) -> bool:
            if not q_lower and is_noise_log_entry(e):
                return False
            if q_lower and q_lower not in e["msg"].lower() and q_lower not in e.get("source", "").lower():
                return False
            if level_upper and e["level"] != level_upper:
                return False
            if source_lower and source_lower not in e.get("source", "").lower():
                return False
            return True

        filtered = [e for e in entries if matches(e)]
        filtered.sort(key=lambda e: e.get("ts", ""), reverse=True)
        return filtered[:limit]

    return app
