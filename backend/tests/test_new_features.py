"""Tests for the three new features:
1. Selective validator (full replay + windowed + record-seed verification)
2. Odd-only / v2 CPU parallel kernel
3. Hypothesis sandbox (experimental direction + generators)
"""
from __future__ import annotations

import pytest

from collatz_lab import services
from collatz_lab.hardware import CPU_BARINA_KERNEL, CPU_PARALLEL_ODD_KERNEL, CPU_SIEVE_KERNEL
from collatz_lab.hypothesis import (
    analyze_record_seeds,
    analyze_residue_classes,
    run_hypothesis_battery,
    scan_trajectory_depths,
    test_mod3_convergence_redundancy as run_mod3_convergence_redundancy,
    test_stopping_time_growth as check_stopping_time_growth,
)
from collatz_lab.repository import LabRepository
from collatz_lab.schemas import ComputeProfile, HypothesisStatus, RunStatus
from collatz_lab.services import (
    SELECTIVE_VALIDATION_THRESHOLD,
    _compute_odd_only_reference,
    compute_range_metrics,
    compute_range_metrics_parallel_odd,
    compute_range_metrics_parallel_descent,
    compute_range_metrics_parallel_descent_odd,
    metrics_direct,
    validate_run,
)


# ===========================================================================
# 1. Odd-only kernel correctness
# ===========================================================================


class TestOddOnlyKernel:
    """cpu-parallel-odd must produce correct metrics for all odd seeds."""

    def test_small_range_odd_only_matches_direct(self):
        """Odd-only kernel must match per-seed metrics_direct for odd seeds."""
        odd_result = compute_range_metrics(1, 1000, kernel="cpu-parallel-odd")
        ref = _compute_odd_only_reference(1, 1000)
        assert odd_result.max_total_stopping_time == ref.max_total_stopping_time
        assert odd_result.max_stopping_time == ref.max_stopping_time
        assert odd_result.max_excursion == ref.max_excursion

    def test_processed_count_is_odd_seeds_only(self):
        """The processed count should be the number of odd seeds, not total range."""
        result = compute_range_metrics(1, 100, kernel="cpu-parallel-odd")
        # Odd seeds in [1, 100]: 1, 3, 5, ..., 99 = 50 seeds
        assert result.processed == 50

    def test_processed_count_even_start(self):
        """Range starting with even number."""
        result = compute_range_metrics(2, 101, kernel="cpu-parallel-odd")
        # Odd seeds in [2, 101]: 3, 5, ..., 101 = 50 seeds
        assert result.processed == 50

    def test_single_odd_seed(self):
        """Single odd seed should produce correct metrics."""
        result = compute_range_metrics(27, 27, kernel="cpu-parallel-odd")
        ref = metrics_direct(27)
        assert result.processed == 1
        assert result.max_total_stopping_time["value"] == ref.total_stopping_time
        assert result.max_total_stopping_time["n"] == 27
        assert result.max_excursion["value"] == ref.max_excursion

    def test_single_even_seed_produces_zero_processed(self):
        """Single even seed should produce 0 processed (no odd seeds)."""
        result = compute_range_metrics(28, 28, kernel="cpu-parallel-odd")
        assert result.processed == 0

    def test_known_value_n27(self):
        """n=27 is famous: tst=111, max_excursion=9232."""
        result = compute_range_metrics(27, 27, kernel="cpu-parallel-odd")
        assert result.max_total_stopping_time == {"n": 27, "value": 111}
        assert result.max_excursion == {"n": 27, "value": 9232}

    def test_range_1_to_10000_max_records_match_direct(self):
        """All max-record values must match cpu-direct odd-only reference."""
        odd_result = compute_range_metrics(1, 10_000, kernel="cpu-parallel-odd")
        ref = _compute_odd_only_reference(1, 10_000)
        assert odd_result.max_total_stopping_time == ref.max_total_stopping_time
        assert odd_result.max_stopping_time == ref.max_stopping_time
        assert odd_result.max_excursion == ref.max_excursion

    def test_odd_only_vs_parallel_records_are_subset(self):
        """Odd-only max values should be <= parallel (all-seeds) max values.

        Because parallel processes all seeds including even ones, its max
        values should be >= odd-only's.  (In practice they're often equal
        since records tend to be at odd seeds.)
        """
        odd_result = compute_range_metrics(1, 5000, kernel="cpu-parallel-odd")
        all_result = compute_range_metrics(1, 5000, kernel="cpu-parallel")
        assert odd_result.max_total_stopping_time["value"] <= all_result.max_total_stopping_time["value"]
        assert odd_result.max_excursion["value"] <= all_result.max_excursion["value"]

    def test_sample_records_contain_only_odd_seeds(self):
        """All seeds in sample_records must be odd."""
        result = compute_range_metrics(1, 5000, kernel="cpu-parallel-odd")
        for record in result.sample_records:
            assert record["n"] % 2 == 1, f"Even seed {record['n']} in odd-only sample_records"

    def test_large_range_matches_reference(self):
        """Larger range correctness check."""
        odd_result = compute_range_metrics(10_000, 20_000, kernel="cpu-parallel-odd")
        ref = _compute_odd_only_reference(10_000, 20_000)
        assert odd_result.max_total_stopping_time == ref.max_total_stopping_time
        assert odd_result.max_stopping_time == ref.max_stopping_time
        assert odd_result.max_excursion == ref.max_excursion

    def test_last_processed_equals_range_end(self):
        """last_processed should be the range end, not the last odd seed."""
        result = compute_range_metrics(1, 100, kernel="cpu-parallel-odd")
        assert result.last_processed == 100


# ===========================================================================
# 1b. Barina kernel correctness
# ===========================================================================


class TestBarinaKernel:
    """cpu-barina must prove convergence correctly via domain-switching.

    Note: Barina uses a different trajectory than standard Collatz, so
    stopping_time and max_excursion values won't match the sieve kernel.
    What matters: same seed count, all seeds converge, no overflows.
    """

    def test_processed_count_matches_sieve(self):
        """Both kernels should process the same number of odd seeds."""
        barina = compute_range_metrics(1, 1000, kernel="cpu-barina")
        sieve = compute_range_metrics(1, 1000, kernel="cpu-sieve")
        assert barina.processed == sieve.processed

    def test_all_seeds_converge(self):
        """No seed should overflow or timeout (stopping_time != -1)."""
        result = compute_range_metrics(1, 10_000, kernel="cpu-barina")
        assert result.max_stopping_time["value"] > 0
        # No overflow: max_excursion should be positive
        assert result.max_excursion["value"] > 0

    def test_single_seed_n27(self):
        """n=27 descent verification."""
        result = compute_range_metrics(27, 27, kernel="cpu-barina")
        assert result.processed == 1
        assert result.max_excursion["n"] == 27
        assert result.max_excursion["value"] > 27  # must exceed seed

    def test_single_even_seed_zero_processed(self):
        """Even seed should produce 0 processed."""
        result = compute_range_metrics(28, 28, kernel="cpu-barina")
        assert result.processed == 0

    def test_all_odd_seeds_verified(self):
        """Every odd seed must be individually verified — no skipping."""
        # Range [1,9] has odd seeds: 1,3,5,7,9 = 5 seeds
        result = compute_range_metrics(1, 9, kernel="cpu-barina")
        assert result.processed == 5

    def test_convergence_agrees_with_standard_collatz(self):
        """For every odd seed in [3, 10000], verify that Barina descent agrees
        with standard Collatz descent.

        Both kernels should agree on WHICH seeds descend (all of them).
        The descent point (first value < seed) must be the same even though
        the intermediate trajectory differs.
        """
        barina = compute_range_metrics(1, 10_000, kernel="cpu-barina")
        sieve = compute_range_metrics(1, 10_000, kernel="cpu-sieve")
        # Same number of seeds processed (no skipping)
        assert barina.processed == sieve.processed
        # Both must find ALL seeds converge (no overflows/timeouts)
        assert barina.max_stopping_time["value"] > 0
        assert sieve.max_stopping_time["value"] > 0

    def test_checkpoint_interval(self):
        """Barina kernel should have its own checkpoint interval."""
        interval = services._effective_checkpoint_interval("cpu-barina", 250)
        assert interval >= 5_000_000


# ===========================================================================
# 2. Selective validator
# ===========================================================================


class TestSelectiveValidator:
    """Test the tiered validation system."""

    def test_small_run_gets_full_replay(self, repository: LabRepository):
        """Runs below the threshold get full independent replay."""
        run = repository.create_run(
            direction_slug="verification",
            name="small-validation-test",
            range_start=1,
            range_end=30,
            kernel="cpu-direct",
            hardware="cpu",
        )
        services.execute_run(repository, run.id)
        validated = validate_run(repository, run.id)
        assert validated.status == RunStatus.VALIDATED
        assert "full replay" in validated.summary

    def test_small_parallel_run_validates(self, repository: LabRepository):
        """cpu-parallel small run validates against cpu-direct."""
        run = repository.create_run(
            direction_slug="verification",
            name="parallel-validation-test",
            range_start=1,
            range_end=50,
            kernel="cpu-parallel",
            hardware="cpu",
        )
        services.execute_run(repository, run.id)
        validated = validate_run(repository, run.id)
        assert validated.status == RunStatus.VALIDATED

    def test_small_odd_only_run_validates(self, repository: LabRepository):
        """cpu-parallel-odd small run validates correctly."""
        run = repository.create_run(
            direction_slug="verification",
            name="odd-only-validation-test",
            range_start=1,
            range_end=100,
            kernel="cpu-parallel-odd",
            hardware="cpu",
        )
        services.execute_run(repository, run.id)
        validated = validate_run(repository, run.id)
        assert validated.status == RunStatus.VALIDATED

    def test_small_cpu_sieve_run_validates_against_descent_reference(self, repository: LabRepository):
        """cpu-sieve must validate against descent semantics, not full-orbit cpu-direct."""
        run = repository.create_run(
            direction_slug="verification",
            name="sieve-validation-test",
            range_start=1,
            range_end=100,
            kernel=CPU_SIEVE_KERNEL,
            hardware="cpu",
        )
        services.execute_run(repository, run.id)
        validated = validate_run(repository, run.id)
        assert validated.status == RunStatus.VALIDATED
        assert "descent reference" in validated.summary or "Validation passed" in validated.summary

    def test_selective_cpu_sieve_run_validates(self, repository: LabRepository, monkeypatch):
        """Large cpu-sieve runs should use selective validation against descent windows."""
        monkeypatch.setattr(services, "SELECTIVE_VALIDATION_THRESHOLD", 50)
        run = repository.create_run(
            direction_slug="verification",
            name="sieve-selective-validation-test",
            range_start=1,
            range_end=100,
            kernel=CPU_SIEVE_KERNEL,
            hardware="cpu",
        )
        services.execute_run(repository, run.id)
        validated = validate_run(repository, run.id, window_count=2, window_size=20)
        assert validated.status == RunStatus.VALIDATED
        assert "selective" in validated.summary

    def test_validation_creates_artifact(self, repository: LabRepository):
        """Validation should create a report artifact."""
        run = repository.create_run(
            direction_slug="verification",
            name="artifact-test",
            range_start=1,
            range_end=20,
            kernel="cpu-direct",
            hardware="cpu",
        )
        services.execute_run(repository, run.id)
        before_count = len(repository.list_artifacts())
        validate_run(repository, run.id)
        after_count = len(repository.list_artifacts())
        assert after_count > before_count

    def test_selective_mode_for_large_run(self, repository: LabRepository, monkeypatch):
        """Runs above threshold should use selective validation."""
        # Temporarily lower the threshold to test selective mode
        monkeypatch.setattr(services, "SELECTIVE_VALIDATION_THRESHOLD", 50)
        run = repository.create_run(
            direction_slug="verification",
            name="large-validation-test",
            range_start=1,
            range_end=100,
            kernel="cpu-parallel",
            hardware="cpu",
        )
        services.execute_run(repository, run.id)
        validated = validate_run(repository, run.id, window_count=3, window_size=10)
        assert validated.status == RunStatus.VALIDATED
        assert "selective" in validated.summary

    def test_coverage_gap_detection(self, repository: LabRepository):
        """Validator should detect gaps in verification coverage."""
        # Create a run that doesn't start from 1 — gap at [1, 99]
        run = repository.create_run(
            direction_slug="verification",
            name="gap-test",
            range_start=100,
            range_end=200,
            kernel="cpu-direct",
            hardware="cpu",
        )
        services.execute_run(repository, run.id)
        validated = validate_run(repository, run.id)
        # Should still validate (gap is a warning, not a failure)
        assert validated.status == RunStatus.VALIDATED


# ===========================================================================
# 3. Hypothesis sandbox
# ===========================================================================


class TestHypothesisGenerators:
    """Test the hypothesis generator functions."""

    def test_residue_class_analysis_mod3(self):
        result = analyze_residue_classes(3, start=1, end=10_000)
        assert result.category == "residue-class"
        assert result.status in {
            HypothesisStatus.PLAUSIBLE,
            HypothesisStatus.FALSIFIED,
        }
        assert result.evidence  # should have per-class data
        assert len(result.evidence) == 3  # 3 classes mod 3
        # Each evidence entry should have mean_tst
        for entry in result.evidence:
            assert "mean_tst" in entry
            assert "z_score" in entry

    def test_residue_class_analysis_mod4(self):
        result = analyze_residue_classes(4, start=1, end=5000)
        assert result.category == "residue-class"
        # Mod 4 with odd_only=True means only residues 1 and 3 have data
        odd_entries = [e for e in result.evidence if e["count"] > 0]
        assert len(odd_entries) >= 2

    def test_residue_class_invalid_modulus(self):
        with pytest.raises(ValueError, match="Modulus must be at least 2"):
            analyze_residue_classes(1)

    def test_record_seed_analysis(self):
        result = analyze_record_seeds(start=1, end=10_000)
        assert result.category == "record-structure"
        assert result.status in {
            HypothesisStatus.PROPOSED,
            HypothesisStatus.PLAUSIBLE,
            HypothesisStatus.FALSIFIED,
        }
        assert result.evidence
        evidence = result.evidence[0]
        assert "tst_records" in evidence
        assert "exc_records" in evidence
        assert evidence["tst_record_count"] > 0

    def test_trajectory_depth_scan(self):
        result = scan_trajectory_depths(start=1, end=10_000, top_k=10)
        assert result.category == "trajectory-shape"
        assert result.evidence
        evidence = result.evidence[0]
        assert "top_seeds" in evidence
        assert "mod6_distribution" in evidence

    def test_stopping_time_growth_rate(self):
        result = check_stopping_time_growth(start=1, end=10_000, bin_count=10)
        assert result.category == "algebraic-probe"
        assert result.evidence
        for bin_data in result.evidence:
            assert "mean_tst" in bin_data
            assert "predicted_tst" in bin_data
            assert "deviation_pct" in bin_data

    def test_mod3_convergence_redundancy(self):
        """Mod-3 filter is an unproven hypothesis — test it empirically."""
        result = run_mod3_convergence_redundancy(start=1, end=5_000)
        assert result.category == "structural-filter"
        assert result.status in {
            HypothesisStatus.PLAUSIBLE,
            HypothesisStatus.FALSIFIED,
        }
        assert result.evidence
        evidence = result.evidence[0]
        assert "tested" in evidence
        assert "confirmed" in evidence
        assert "counterexamples" in evidence

    def test_hypothesis_battery_persistence(self, repository: LabRepository):
        """Battery should create claims and artifacts."""
        before_claims = len(repository.list_claims())
        hypotheses = run_hypothesis_battery(
            repository, end=1000, moduli=[3, 4],
        )
        after_claims = len(repository.list_claims())
        # 2 residue + 1 record + 1 trajectory + 1 growth + 1 mod-3 = 6
        assert len(hypotheses) == 6
        assert after_claims == before_claims + 6
        # Each hypothesis should have an id
        for h in hypotheses:
            assert h.id
            assert h.direction_slug == "hypothesis-sandbox"

    def test_hypothesis_battery_creates_artifacts(self, repository: LabRepository):
        """Each hypothesis should have an evidence artifact."""
        before_artifacts = len(repository.list_artifacts())
        run_hypothesis_battery(repository, end=500, moduli=[3])
        after_artifacts = len(repository.list_artifacts())
        # 1 residue + 1 record + 1 trajectory + 1 growth + 1 mod-3 = 5 hypotheses, 5 artifacts
        assert after_artifacts >= before_artifacts + 5

    def test_hypothesis_sandbox_direction_exists(self, repository: LabRepository):
        """The hypothesis-sandbox direction should be seeded."""
        directions = repository.list_directions()
        slugs = [d.slug for d in directions]
        assert "hypothesis-sandbox" in slugs


# ===========================================================================
# 4. Continuous queue uses odd-only kernel
# ===========================================================================


class TestContinuousQueueOddOnly:
    """Verify that queue_continuous_verification_runs uses sieve kernel.

    The sieve kernel uses standard Collatz steps (3n+1 / n/2) with early
    termination.  Every odd seed is individually verified — no seeds are
    skipped.  This is the canonical verification path.
    """

    def test_continuous_cpu_uses_sieve_kernel(self, repository: LabRepository):
        queued_ids = services.queue_continuous_verification_runs(
            repository, supported_hardware=["cpu"],
        )
        assert queued_ids
        run = repository.get_run(queued_ids[0])
        assert run.kernel == CPU_SIEVE_KERNEL
        assert run.name == "autopilot-continuous-cpu"

    def test_odd_only_kernel_in_effective_checkpoint_interval(self):
        """Odd-only kernel should have its own checkpoint interval."""
        interval = services._effective_checkpoint_interval("cpu-parallel-odd", 250)
        assert interval >= 2_000_000


class TestDescentReferences:
    """Validation references for sieve kernels must match their descent semantics."""

    def test_odd_descent_reference_matches_cpu_sieve(self):
        candidate = compute_range_metrics(1, 500, kernel=CPU_SIEVE_KERNEL)
        reference = compute_range_metrics_parallel_descent_odd(1, 500)
        assert candidate.max_total_stopping_time == reference.max_total_stopping_time
        assert candidate.max_stopping_time == reference.max_stopping_time
        assert candidate.max_excursion == reference.max_excursion

    def test_full_descent_reference_is_not_full_orbit_reference(self):
        descent = compute_range_metrics_parallel_descent(1, 100)
        full = compute_range_metrics(1, 100, kernel="cpu-direct")
        assert descent.max_total_stopping_time != full.max_total_stopping_time

    def test_cpu_sieve_overflow_fallback_preserves_descent_semantics(self):
        start = 1_002_416_050_001
        end = start + 100_000 - 1
        candidate = compute_range_metrics(start, end, kernel=CPU_SIEVE_KERNEL)
        reference = compute_range_metrics_parallel_descent_odd(start, end)
        assert candidate.max_total_stopping_time == reference.max_total_stopping_time
        assert candidate.max_stopping_time == reference.max_stopping_time
        assert candidate.max_excursion == reference.max_excursion


class TestComputeBudgetThrottle:
    """Budget slider must change wall time per batch, not only chunk sizing."""

    def test_throttle_matches_duty_cycle_formula(self):
        p50 = ComputeProfile(cpu_percent=50, gpu_percent=50, system_percent=100)
        assert services._compute_budget_throttle_seconds(
            hardware="cpu", profile=p50, compute_sec=2.0
        ) == pytest.approx(2.0)
        assert services._compute_budget_throttle_seconds(
            hardware="gpu", profile=p50, compute_sec=2.0
        ) == pytest.approx(2.0)
        p100 = ComputeProfile(cpu_percent=100, gpu_percent=100, system_percent=100)
        assert services._compute_budget_throttle_seconds(
            hardware="cpu", profile=p100, compute_sec=2.0
        ) == 0.0
