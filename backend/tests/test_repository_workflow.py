from __future__ import annotations

from collatz_lab.repository import LabRepository
from collatz_lab.schemas import MapVariant, ReviewRubric, RunStatus, SourceReviewMode, SourceStatus
from collatz_lab.services import execute_run, generate_report, validate_run


def test_repository_seeds_default_directions(repository: LabRepository):
    directions = repository.list_directions()
    assert {direction.slug for direction in directions} == {
        "verification",
        "inverse-tree-parity",
        "lemma-workspace",
        "two-adic-v2",
        "hypothesis-sandbox",
    }


def test_full_run_claim_and_report_flow(repository: LabRepository):
    task = repository.create_task(
        direction_slug="verification",
        title="Baseline sweep",
        kind="experiment",
        description="Sweep small interval",
    )
    assert task.id.startswith("COL-")

    run = repository.create_run(
        direction_slug="verification",
        name="baseline",
        range_start=1,
        range_end=50,
    )
    run = execute_run(repository, run.id, checkpoint_interval=10)
    assert run.status == RunStatus.COMPLETED
    assert run.metrics["processed"] == 50

    run = validate_run(repository, run.id)
    assert run.status == RunStatus.VALIDATED

    claim = repository.create_claim(
        direction_slug="lemma-workspace",
        title="Small interval pattern",
        statement="Validation remains consistent on the small baseline interval.",
        dependencies=[run.id],
    )
    repository.link_claim_run(claim_id=claim.id, run_id=run.id, relation="supports")
    review = repository.review_direction("lemma-workspace")
    assert review.direction.status in {"active", "promising"}

    report_path = generate_report(repository)
    assert report_path.exists()

    links = repository.list_claim_run_links()
    assert any(link.claim_id == claim.id and link.run_id == run.id for link in links)

    claim_artifact = next(artifact for artifact in repository.list_artifacts() if artifact.claim_id == claim.id)
    content_payload = repository.read_artifact_content(claim_artifact.id)
    assert "Small interval pattern" in content_payload["content"]


def test_source_review_refreshes_note_and_snapshot(repository: LabRepository):
    source = repository.create_source(
        direction_slug="lemma-workspace",
        title="External attempt",
        summary="Initial intake note.",
        fallacy_tags=["empirical-not-proof"],
    )

    reviewed = repository.update_source_review(
        source.id,
        review_status=SourceStatus.FLAGGED,
        map_variant=MapVariant.STANDARD,
        notes="Missing universal step.",
        fallacy_tags=["unchecked-generalization", "variant-confusion"],
        rubric=ReviewRubric(
            defines_map_variant=True,
            distinguishes_empirical_from_proof=True,
            proves_cycle_exclusion=False,
        ),
    )

    assert reviewed.review_status == SourceStatus.FLAGGED
    assert reviewed.map_variant == MapVariant.STANDARD
    assert reviewed.fallacy_tags == ["unchecked-generalization", "variant-confusion"]

    source_note = repository.settings.research_dir / "sources" / f"{source.id}.md"
    assert source_note.exists()
    note_text = source_note.read_text(encoding="utf-8")
    assert "Map Variant: standard" in note_text
    assert "unchecked-generalization" in note_text
    assert "variant-confusion" in note_text

    review_snapshots = list((repository.settings.artifacts_dir / "reviews").glob(f"{source.id}-*.md"))
    assert review_snapshots

    history = repository.list_source_reviews(source.id)
    assert len(history) == 2
    assert history[0].mode == SourceReviewMode.MANUAL
    assert history[0].review_status == SourceStatus.FLAGGED
    assert history[1].mode == SourceReviewMode.INTAKE
    assert history[1].review_status == SourceStatus.INTAKE


def test_register_worker_requeues_orphaned_running_runs(repository: LabRepository):
    run = repository.create_run(
        direction_slug="verification",
        name="resume-me",
        range_start=1,
        range_end=100,
    )
    repository.update_run(
        run.id,
        status=RunStatus.RUNNING,
        checkpoint={"next_value": 51, "last_processed": 50},
        metrics={"processed": 50, "last_processed": 50},
        summary="Halfway through the interval.",
        started_at="2026-03-21T10:00:00+00:00",
    )

    worker = repository.register_worker(
        name="recovering-worker",
        role="compute-agent",
        hardware="cpu",
        capabilities=[],
    )

    recovered_run = repository.get_run(run.id)
    assert worker.status == "idle"
    assert recovered_run.status == RunStatus.QUEUED
    assert "Recovered after worker restart" in recovered_run.summary


def test_delete_source_removes_reviews_and_files(repository: LabRepository):
    source = repository.create_source(
        direction_slug="lemma-workspace",
        title="Delete me",
        summary="Temporary local source.",
    )
    repository.update_source_review(
        source.id,
        review_status=SourceStatus.FLAGGED,
        notes="Temporary review snapshot.",
    )

    result = repository.delete_source(source.id)

    assert result["deleted"] is True
    assert source.id == result["id"]
    assert not (repository.settings.research_dir / "sources" / f"{source.id}.md").exists()
    assert not list((repository.settings.artifacts_dir / "reviews").glob(f"{source.id}-*.md"))
    assert not any(item.id == source.id for item in repository.list_sources())
