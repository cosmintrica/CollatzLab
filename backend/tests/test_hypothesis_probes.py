"""Lightweight tests for new hypothesis probes (stratified residue, glide, scalability)."""

from __future__ import annotations

from collatz_lab.hypothesis import (
    _orbit_odd_even_counts,
    analyze_glide_structure,
    analyze_residue_classes_stratified,
    metrics_direct,
    run_battery_scalability_report,
)
from collatz_lab.repository import LabRepository
from collatz_lab.metrics_sot import metrics_direct as metrics_direct_sot


def test_orbit_odd_even_counts_matches_total_stopping_time():
    for n in (1, 2, 3, 7, 27, 97):
        total, odd_s, even_s = _orbit_odd_even_counts(n)
        m = metrics_direct_sot(n)
        assert total == m.total_stopping_time
        assert odd_s + even_s == total


def test_stratified_residue_small_range_returns_result():
    r = analyze_residue_classes_stratified(4, 1, 2_000, bin_count=6)
    assert r.title
    assert r.evidence
    assert r.evidence[0].get("report_meta", {}).get("probe_kind") == "residue_class_stratified_log2"
    assert r.category == "residue-class"


def test_glide_structure_small_range():
    r = analyze_glide_structure(1, 3_000, sample_cap=400, bootstrap_reps=200)
    assert r.category == "orbit-structure"
    assert r.evidence
    ev0 = r.evidence[0]
    assert ev0.get("report_meta", {}).get("probe_kind") == "glide_odd_fraction_terras"
    assert "terras_odd_fraction" in ev0
    assert "global_bootstrap_ci95_low" in ev0
    assert ev0["sampled_seeds"] >= 1


def test_battery_scalability_report_fast_endpoints():
    rep = run_battery_scalability_report(endpoints=[800, 2_500])
    assert "by_scale" in rep
    assert "800" in rep["by_scale"]
    assert "2500" in rep["by_scale"]
    assert "stratified_mod8" in rep["by_scale"]["800"]
    assert isinstance(rep["status_flips"], list)
    assert rep.get("stability_verdict")
    assert isinstance(rep.get("summary_lines"), list)
    assert rep.get("report_meta", {}).get("probe_kind") == "battery_scalability_meta"


def test_battery_scalability_persist_writes_artifact(repository: LabRepository):
    before = len(repository.list_artifacts())
    rep = run_battery_scalability_report(
        endpoints=[500, 2_000],
        repository=repository,
        persist=True,
    )
    assert rep["summary_lines"]
    assert len(repository.list_artifacts()) >= before + 1


def test_promising_followup_task_idempotent(repository: LabRepository):
    from collatz_lab.hypothesis import enqueue_sandbox_promising_followup_task

    claim = repository.create_claim(
        direction_slug="hypothesis-sandbox",
        title="Synthetic promising probe",
        statement="Test",
        owner="test",
    )
    repository.update_claim_status(claim.id, "promising")
    matching = [
        t
        for t in repository.list_tasks()
        if "[sandbox-promising-followup]" in (t.description or "")
        and claim.id in (t.description or "")
    ]
    assert len(matching) == 1
    t2 = enqueue_sandbox_promising_followup_task(
        repository, claim_id=claim.id, claim_title=claim.title
    )
    assert t2 is None


def test_hypothesis_imports_metrics_direct_from_sot():
    assert metrics_direct is metrics_direct_sot
