"""Hypothesis Sandbox — experimental track for novel mathematical exploration.

This module generates and tests falsifiable hypotheses about the Collatz
conjecture.  All outputs remain in the ``hypothesis-sandbox`` direction
and never promote directly to supported/validated claims.

Categories
----------
- ``residue-class``    : statistical patterns in stopping times by residue class
- ``record-structure`` : structural properties of record-breaking seeds
- ``trajectory-shape`` : orbit shape classification and prediction
- ``algebraic-probe``  : searching for algebraic relationships
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from .repository import LabRepository, utc_now
from .schemas import ArtifactKind, Hypothesis, HypothesisStatus
from .services import metrics_direct


DIRECTION_SLUG = "hypothesis-sandbox"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ResidueClassStats:
    residue: int
    count: int
    mean_tst: float
    stddev_tst: float
    mean_excursion: float
    max_tst: int
    max_tst_seed: int
    max_excursion: int
    max_excursion_seed: int


@dataclass
class HypothesisResult:
    title: str
    statement: str
    category: str
    status: HypothesisStatus
    test_methodology: str
    test_range: str
    evidence: list[dict]
    falsification: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# 1. Residue class stopping time analyzer
# ---------------------------------------------------------------------------

def analyze_residue_classes(
    modulus: int,
    start: int = 1,
    end: int = 100_000,
    *,
    odd_only: bool = True,
) -> HypothesisResult:
    """Compute mean total stopping time by residue class mod ``modulus``.

    Identifies residue classes with statistically anomalous stopping times
    and generates a testable hypothesis about the pattern.
    """
    if modulus < 2:
        raise ValueError("Modulus must be at least 2.")

    # Collect per-class data: each entry is (seed, value) so the seed is
    # always recoverable without index arithmetic.
    class_tst: dict[int, list[tuple[int, int]]] = {r: [] for r in range(modulus)}
    class_exc: dict[int, list[tuple[int, int]]] = {r: [] for r in range(modulus)}

    step = 2 if odd_only else 1
    first = start if (not odd_only or start & 1) else start + 1
    for n in range(first, end + 1, step):
        m = metrics_direct(n)
        r = n % modulus
        class_tst[r].append((n, m.total_stopping_time))
        class_exc[r].append((n, m.max_excursion))

    # Build stats
    stats: list[ResidueClassStats] = []
    for r in range(modulus):
        tst_pairs = class_tst[r]
        exc_pairs = class_exc[r]
        if not tst_pairs:
            continue
        tst_values = [v for _, v in tst_pairs]
        exc_values = [v for _, v in exc_pairs]
        best_tst = max(tst_pairs, key=lambda p: p[1])
        best_exc = max(exc_pairs, key=lambda p: p[1])
        stats.append(ResidueClassStats(
            residue=r,
            count=len(tst_values),
            mean_tst=statistics.mean(tst_values),
            stddev_tst=statistics.stdev(tst_values) if len(tst_values) > 1 else 0.0,
            mean_excursion=statistics.mean(exc_values),
            max_tst=best_tst[1],
            max_tst_seed=best_tst[0],
            max_excursion=best_exc[1],
            max_excursion_seed=best_exc[0],
        ))

    if not stats:
        return HypothesisResult(
            title=f"Residue class analysis mod {modulus}",
            statement="Insufficient data to form hypothesis.",
            category="residue-class",
            status=HypothesisStatus.FALSIFIED,
            test_methodology="N/A",
            test_range=f"{start}-{end}",
            evidence=[],
        )

    # Find anomalous classes: those with mean_tst > 1.5 stddev above grand mean
    grand_mean = statistics.mean(s.mean_tst for s in stats)
    grand_stddev = statistics.stdev(s.mean_tst for s in stats) if len(stats) > 1 else 0.0

    high_classes = [
        s for s in stats
        if grand_stddev > 0 and (s.mean_tst - grand_mean) > 1.5 * grand_stddev
    ]
    low_classes = [
        s for s in stats
        if grand_stddev > 0 and (grand_mean - s.mean_tst) > 1.5 * grand_stddev
    ]

    evidence = [
        {
            "residue": s.residue,
            "count": s.count,
            "mean_tst": round(s.mean_tst, 2),
            "stddev_tst": round(s.stddev_tst, 2),
            "mean_excursion": round(s.mean_excursion, 2),
            "z_score": round((s.mean_tst - grand_mean) / grand_stddev, 3) if grand_stddev > 0 else 0.0,
        }
        for s in stats
    ]

    if high_classes:
        high_residues = [s.residue for s in high_classes]
        statement = (
            f"Seeds ≡ {high_residues} (mod {modulus}) have significantly higher "
            f"mean total stopping time ({', '.join(f'{s.mean_tst:.1f}' for s in high_classes)}) "
            f"compared to grand mean {grand_mean:.1f} (z > 1.5σ) over [{start}, {end}]."
        )
        status = HypothesisStatus.PLAUSIBLE
    elif low_classes:
        low_residues = [s.residue for s in low_classes]
        statement = (
            f"Seeds ≡ {low_residues} (mod {modulus}) converge significantly faster "
            f"(mean TST {', '.join(f'{s.mean_tst:.1f}' for s in low_classes)}) "
            f"compared to grand mean {grand_mean:.1f} (z < -1.5σ) over [{start}, {end}]."
        )
        status = HypothesisStatus.PLAUSIBLE
    else:
        statement = (
            f"No residue class mod {modulus} shows statistically significant "
            f"deviation in mean total stopping time over [{start}, {end}]. "
            f"Grand mean = {grand_mean:.1f}, σ = {grand_stddev:.2f}."
        )
        status = HypothesisStatus.FALSIFIED

    return HypothesisResult(
        title=f"Residue class stopping time distribution mod {modulus}",
        statement=statement,
        category="residue-class",
        status=status,
        test_methodology=(
            f"Computed metrics_direct for {'odd' if odd_only else 'all'} seeds in "
            f"[{start}, {end}], grouped by residue mod {modulus}. "
            f"Flagged classes with mean TST > 1.5σ from grand mean."
        ),
        test_range=f"{start}-{end}",
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# 2. Record-breaking seed structure analyzer
# ---------------------------------------------------------------------------

def analyze_record_seeds(
    start: int = 1,
    end: int = 100_000,
) -> HypothesisResult:
    """Analyze structural properties of record-breaking seeds.

    Looks at binary length, Hamming weight, and v2 of record-breaking
    seeds (for total stopping time and max excursion) and tests whether
    they share common structural properties.
    """
    records_tst: list[dict] = []
    records_exc: list[dict] = []
    current_max_tst = -1
    current_max_exc = -1

    for n in range(max(1, start), end + 1):
        m = metrics_direct(n)
        if m.total_stopping_time > current_max_tst:
            current_max_tst = m.total_stopping_time
            bit_len = n.bit_length()
            hamming = bin(n).count("1")
            v2 = 0
            temp = n
            while temp > 0 and temp & 1 == 0:
                v2 += 1
                temp >>= 1
            records_tst.append({
                "n": n,
                "tst": m.total_stopping_time,
                "bit_length": bit_len,
                "hamming_weight": hamming,
                "hamming_density": round(hamming / bit_len, 4),
                "v2": v2,
            })
        if m.max_excursion > current_max_exc:
            current_max_exc = m.max_excursion
            bit_len = n.bit_length()
            hamming = bin(n).count("1")
            records_exc.append({
                "n": n,
                "excursion": m.max_excursion,
                "bit_length": bit_len,
                "hamming_weight": hamming,
                "hamming_density": round(hamming / bit_len, 4),
            })

    evidence = {
        "tst_records": records_tst[-20:],  # last 20 record-breakers
        "exc_records": records_exc[-20:],
        "tst_record_count": len(records_tst),
        "exc_record_count": len(records_exc),
    }

    # Analyze Hamming density distribution of TST record seeds
    if len(records_tst) >= 5:
        densities = [r["hamming_density"] for r in records_tst]
        mean_density = statistics.mean(densities)
        # For random odd numbers, expected Hamming density ≈ 0.5
        if mean_density > 0.55:
            statement = (
                f"Record-breaking TST seeds in [{start}, {end}] have mean Hamming "
                f"density {mean_density:.3f} (above random expectation ~0.5). "
                f"High bit density may correlate with longer orbits."
            )
            status = HypothesisStatus.PLAUSIBLE
        elif mean_density < 0.45:
            statement = (
                f"Record-breaking TST seeds in [{start}, {end}] have mean Hamming "
                f"density {mean_density:.3f} (below random expectation ~0.5). "
                f"Sparse binary representations may indicate longer orbits."
            )
            status = HypothesisStatus.PLAUSIBLE
        else:
            statement = (
                f"Record-breaking TST seeds in [{start}, {end}] have mean Hamming "
                f"density {mean_density:.3f}, close to random expectation. "
                f"No structural signal detected in binary representation."
            )
            status = HypothesisStatus.FALSIFIED
    else:
        statement = f"Too few record seeds ({len(records_tst)}) to detect structural patterns."
        status = HypothesisStatus.PROPOSED

    # Check if records cluster at specific residue classes mod 4
    if records_tst:
        mod4_counts = {r: 0 for r in range(4)}
        for rec in records_tst:
            mod4_counts[rec["n"] % 4] += 1
        dominant_class = max(mod4_counts, key=lambda r: mod4_counts[r])
        dominant_frac = mod4_counts[dominant_class] / len(records_tst)
        if dominant_frac > 0.5:
            statement += (
                f" Additionally, {dominant_frac:.0%} of TST records are ≡ {dominant_class} (mod 4)."
            )

    return HypothesisResult(
        title=f"Structure of record-breaking seeds in [{start}, {end}]",
        statement=statement,
        category="record-structure",
        status=status,
        test_methodology=(
            f"Scanned [{start}, {end}], tracked seeds that set new records for "
            f"total stopping time and max excursion. Analyzed binary length, "
            f"Hamming weight/density, v2 valuation, and residue class distribution."
        ),
        test_range=f"{start}-{end}",
        evidence=[evidence],
    )


# ---------------------------------------------------------------------------
# 3. Trajectory depth scanner — find seeds with extreme excursions
# ---------------------------------------------------------------------------

def scan_trajectory_depths(
    start: int = 1,
    end: int = 50_000,
    *,
    top_k: int = 20,
) -> HypothesisResult:
    """Find seeds whose orbits reach unusually high values relative to
    their magnitude and analyze the algebraic structure of those seeds.

    Tests the hypothesis: seeds that achieve excursion ratio
    max_excursion / n > threshold share structural properties.
    """
    entries: list[dict] = []

    for n in range(max(1, start), end + 1):
        if n & 1 == 0:
            continue  # only odd seeds are interesting
        m = metrics_direct(n)
        ratio = m.max_excursion / n if n > 0 else 0
        entries.append({
            "n": n,
            "tst": m.total_stopping_time,
            "excursion": m.max_excursion,
            "ratio": ratio,
            "log_ratio": round(math.log2(ratio) if ratio > 0 else 0, 3),
        })

    # Sort by excursion ratio, take top_k
    entries.sort(key=lambda e: e["ratio"], reverse=True)
    top = entries[:top_k]

    if not top:
        return HypothesisResult(
            title="Trajectory depth scan",
            statement="No data.",
            category="trajectory-shape",
            status=HypothesisStatus.FALSIFIED,
            test_methodology="N/A",
            test_range=f"{start}-{end}",
            evidence=[],
        )

    # Analyze top seeds
    top_seeds = [e["n"] for e in top]
    mean_log_ratio = statistics.mean(e["log_ratio"] for e in top)

    # Check if top seeds share a residue class pattern
    mod6_counts: dict[int, int] = {}
    for n in top_seeds:
        r = n % 6
        mod6_counts[r] = mod6_counts.get(r, 0) + 1

    dominant_residue = max(mod6_counts, key=lambda r: mod6_counts[r])
    dominant_frac = mod6_counts[dominant_residue] / len(top_seeds)

    if dominant_frac > 0.5:
        statement = (
            f"Among the top-{top_k} seeds by excursion ratio in [{start}, {end}], "
            f"{dominant_frac:.0%} are ≡ {dominant_residue} (mod 6). "
            f"Mean log₂(excursion/n) = {mean_log_ratio:.2f}. "
            f"This suggests residue class mod 6 may predict extreme excursions."
        )
        status = HypothesisStatus.PLAUSIBLE
    else:
        statement = (
            f"Top-{top_k} seeds by excursion ratio in [{start}, {end}] are "
            f"distributed across mod 6 classes without dominant clustering. "
            f"Mean log₂(excursion/n) = {mean_log_ratio:.2f}. "
            f"No strong residue-class predictor detected."
        )
        status = HypothesisStatus.FALSIFIED

    # Analyze gap pattern between top seeds
    sorted_seeds = sorted(top_seeds)
    if len(sorted_seeds) >= 3:
        gaps = [sorted_seeds[i + 1] - sorted_seeds[i] for i in range(len(sorted_seeds) - 1)]
        mean_gap = statistics.mean(gaps)
        notes = f"Mean gap between top excursion seeds: {mean_gap:.1f}"
    else:
        notes = ""

    evidence = [
        {"top_seeds": [e for e in top[:10]], "mod6_distribution": mod6_counts}
    ]

    return HypothesisResult(
        title=f"Extreme excursion trajectory analysis in [{start}, {end}]",
        statement=statement,
        category="trajectory-shape",
        status=status,
        test_methodology=(
            f"Computed metrics for odd seeds in [{start}, {end}], "
            f"ranked by excursion ratio (max_excursion / n). "
            f"Analyzed top-{top_k} seeds for residue class patterns."
        ),
        test_range=f"{start}-{end}",
        evidence=evidence,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# 4. Stopping time growth rate tester
# ---------------------------------------------------------------------------

def test_stopping_time_growth(
    start: int = 1,
    end: int = 100_000,
    *,
    bin_count: int = 20,
) -> HypothesisResult:
    """Test whether total stopping time grows as C·log(n)^α for some α.

    The heuristic prediction (Lagarias, Wagon) is that the expected
    total stopping time is approximately (6.95212...) · log₂(n).
    We test this against actual data.
    """
    LAGARIAS_CONSTANT = 6.95212

    # Bin seeds by magnitude and compute mean TST per bin
    log_min = math.log2(max(start, 1))
    log_max = math.log2(end)
    bin_width = (log_max - log_min) / bin_count if bin_count > 0 else 1

    bins: list[list[int]] = [[] for _ in range(bin_count)]
    bin_seeds: list[list[int]] = [[] for _ in range(bin_count)]

    for n in range(max(1, start), end + 1):
        if n & 1 == 0:
            continue
        m = metrics_direct(n)
        log_n = math.log2(n)
        bin_idx = min(int((log_n - log_min) / bin_width), bin_count - 1)
        bins[bin_idx].append(m.total_stopping_time)
        bin_seeds[bin_idx].append(n)

    evidence_bins = []
    for i in range(bin_count):
        if not bins[i]:
            continue
        mean_log_n = statistics.mean(math.log2(s) for s in bin_seeds[i])
        mean_tst = statistics.mean(bins[i])
        predicted_tst = LAGARIAS_CONSTANT * mean_log_n
        deviation_pct = ((mean_tst - predicted_tst) / predicted_tst * 100) if predicted_tst > 0 else 0
        evidence_bins.append({
            "bin": i,
            "seed_count": len(bins[i]),
            "mean_log2_n": round(mean_log_n, 2),
            "mean_tst": round(mean_tst, 2),
            "predicted_tst": round(predicted_tst, 2),
            "deviation_pct": round(deviation_pct, 2),
        })

    # Check overall fit
    if evidence_bins:
        max_deviation = max(abs(b["deviation_pct"]) for b in evidence_bins)
        mean_deviation = statistics.mean(abs(b["deviation_pct"]) for b in evidence_bins)

        if max_deviation < 15:
            statement = (
                f"Total stopping time closely follows the heuristic prediction "
                f"TST ≈ {LAGARIAS_CONSTANT:.2f} · log₂(n) over [{start}, {end}]. "
                f"Mean deviation: {mean_deviation:.1f}%, max: {max_deviation:.1f}%."
            )
            status = HypothesisStatus.PLAUSIBLE
        elif max_deviation < 30:
            statement = (
                f"Total stopping time approximately follows {LAGARIAS_CONSTANT:.2f} · log₂(n) "
                f"but with moderate deviations (mean: {mean_deviation:.1f}%, max: {max_deviation:.1f}%) "
                f"over [{start}, {end}]. Possible sublogarithmic correction term."
            )
            status = HypothesisStatus.PLAUSIBLE
        else:
            statement = (
                f"Total stopping time deviates significantly from {LAGARIAS_CONSTANT:.2f} · log₂(n) "
                f"(max deviation: {max_deviation:.1f}%) over [{start}, {end}]. "
                f"The heuristic model may need refinement at this scale."
            )
            status = HypothesisStatus.PROPOSED
    else:
        statement = "Insufficient data."
        status = HypothesisStatus.FALSIFIED

    return HypothesisResult(
        title=f"Stopping time growth rate test over [{start}, {end}]",
        statement=statement,
        category="algebraic-probe",
        status=status,
        test_methodology=(
            f"Binned odd seeds in [{start}, {end}] into {bin_count} logarithmic bins. "
            f"Compared mean TST per bin against the Lagarias-Wagon heuristic "
            f"TST ≈ {LAGARIAS_CONSTANT} · log₂(n)."
        ),
        test_range=f"{start}-{end}",
        evidence=evidence_bins,
    )


# ---------------------------------------------------------------------------
# 5. Mod-3 convergence redundancy test
# ---------------------------------------------------------------------------

def test_mod3_convergence_redundancy(
    start: int = 1,
    end: int = 50_000,
) -> HypothesisResult:
    """Test whether seeds n ≡ 2 (mod 3) can be skipped during verification.

    Claim: if n ≡ 2 (mod 3), then n = 3k+2 for some k ≥ 0.  The smaller
    odd seed (2k+1) has an orbit that passes through a value ≥ n, so n's
    convergence is implied by the convergence of (2k+1) < n.

    This is an UNPROVEN hypothesis.  We test it empirically by checking
    whether every seed n ≡ 2 (mod 3) has a smaller odd predecessor in its
    Collatz orbit backtracking tree.

    This test does NOT prove the claim — it only checks for counterexamples
    in the given range.  The claim requires a rigorous mathematical proof
    before it can be used to skip seeds in production verification.
    """
    counterexamples: list[dict] = []
    tested = 0
    confirmed = 0

    for n in range(max(3, start), end + 1):
        if n & 1 == 0:
            continue
        if n % 3 != 2:
            continue

        tested += 1
        # n = 3k+2, so the claimed predecessor is (2k+1)
        k = (n - 2) // 3
        predecessor = 2 * k + 1

        if predecessor < 1:
            counterexamples.append({
                "n": n, "k": k, "predecessor": predecessor,
                "reason": "predecessor < 1",
            })
            continue

        # Verify: does the orbit of predecessor actually pass through
        # or above n?  Walk the standard Collatz orbit of predecessor.
        current = predecessor
        found = False
        for _step in range(10_000):
            if current >= n:
                found = True
                break
            if current <= 1:
                break
            if current & 1:
                current = 3 * current + 1
            else:
                current >>= 1

        if found:
            confirmed += 1
        else:
            counterexamples.append({
                "n": n, "predecessor": predecessor,
                "reason": f"orbit of {predecessor} never reached {n}",
            })

    if counterexamples:
        status = HypothesisStatus.FALSIFIED
        statement = (
            f"Mod-3 skip hypothesis FALSIFIED: {len(counterexamples)} "
            f"counterexample(s) found in [{start}, {end}]. "
            f"Seeds n ≡ 2 (mod 3) CANNOT be safely skipped."
        )
        falsification = f"Counterexamples: {counterexamples[:10]}"
    elif tested == 0:
        status = HypothesisStatus.PROPOSED
        statement = "No seeds tested (range too small or no n ≡ 2 mod 3)."
        falsification = ""
    else:
        status = HypothesisStatus.PLAUSIBLE
        statement = (
            f"All {confirmed}/{tested} odd seeds n ≡ 2 (mod 3) in [{start}, {end}] "
            f"have a smaller predecessor (2k+1) whose orbit reaches n. "
            f"The mod-3 skip hypothesis is plausible but NOT PROVEN. "
            f"A rigorous proof is required before using this for verification."
        )
        falsification = ""

    return HypothesisResult(
        title=f"Mod-3 convergence redundancy test [{start}, {end}]",
        statement=statement,
        category="structural-filter",
        status=status,
        test_methodology=(
            f"For each odd seed n ≡ 2 (mod 3) in [{start}, {end}], computed "
            f"the claimed predecessor (2k+1) where n=3k+2, then walked the "
            f"standard Collatz orbit of that predecessor to check if it reaches n."
        ),
        test_range=f"{start}-{end}",
        evidence=[{
            "tested": tested,
            "confirmed": confirmed,
            "counterexamples": len(counterexamples),
            "counterexample_details": counterexamples[:10],
        }],
        falsification=falsification,
        notes=(
            "This tests the claim that seeds n ≡ 2 (mod 3) are redundant for "
            "verification because their convergence is implied by a smaller seed. "
            "Even if plausible over a finite range, this is NOT a proof. "
            "Do NOT use for production verification without a published proof."
        ),
    )


# ---------------------------------------------------------------------------
# Orchestrator: run all generators and persist results
# ---------------------------------------------------------------------------

def run_hypothesis_battery(
    repository: LabRepository,
    *,
    end: int = 50_000,
    moduli: list[int] | None = None,
) -> list[Hypothesis]:
    """Run a battery of hypothesis generators and persist results.

    Returns the list of generated/updated hypotheses.
    """
    if moduli is None:
        moduli = [3, 4, 6, 8, 12, 16]

    results: list[HypothesisResult] = []

    # 1. Residue class analysis for each modulus
    for mod in moduli:
        results.append(analyze_residue_classes(mod, start=1, end=end))

    # 2. Record seed structure
    results.append(analyze_record_seeds(start=1, end=end))

    # 3. Trajectory depth scan
    results.append(scan_trajectory_depths(start=1, end=end))

    # 4. Stopping time growth rate
    results.append(test_stopping_time_growth(start=1, end=end))

    # 5. Mod-3 convergence redundancy (structural filter hypothesis)
    results.append(test_mod3_convergence_redundancy(start=1, end=min(end, 10_000)))

    # Persist as claims under hypothesis-sandbox
    hypotheses: list[Hypothesis] = []
    for result in results:
        claim = repository.create_claim(
            direction_slug=DIRECTION_SLUG,
            title=result.title,
            statement=result.statement,
            owner="hypothesis-sandbox",
            notes=result.notes,
        )

        # Map hypothesis status to claim status
        claim_status_map = {
            HypothesisStatus.PROPOSED: "idea",
            HypothesisStatus.TESTING: "active",
            HypothesisStatus.PLAUSIBLE: "promising",
            HypothesisStatus.FALSIFIED: "refuted",
        }
        new_status = claim_status_map.get(result.status, "idea")
        repository.update_claim_status(claim.id, new_status)

        # Save evidence as artifact
        evidence_path = (
            repository.settings.artifacts_dir
            / "hypotheses"
            / f"{claim.id}-evidence.json"
        )
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "hypothesis_id": claim.id,
            "category": result.category,
            "status": result.status,
            "test_methodology": result.test_methodology,
            "test_range": result.test_range,
            "evidence": result.evidence,
            "falsification": result.falsification,
        }
        evidence_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        repository.create_artifact(
            kind=ArtifactKind.JSON,
            path=evidence_path,
            claim_id=claim.id,
            metadata={
                "type": "hypothesis-evidence",
                "category": result.category,
                "status": result.status,
            },
        )

        hypotheses.append(Hypothesis(
            id=claim.id,
            direction_slug=DIRECTION_SLUG,
            title=result.title,
            statement=result.statement,
            category=result.category,
            status=result.status,
            test_methodology=result.test_methodology,
            test_range=result.test_range,
            evidence=result.evidence,
            falsification=result.falsification,
            notes=result.notes,
            created_at=claim.created_at,
            updated_at=claim.updated_at,
        ))

    return hypotheses
