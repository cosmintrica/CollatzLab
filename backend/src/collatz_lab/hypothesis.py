"""Hypothesis Sandbox — experimental track for novel mathematical exploration.

This module generates and tests **falsifiable** hypotheses about the Collatz
conjecture.  We explicitly value *refutation* and structural stress-tests,
not only patterns that appear to support convergence.  All outputs remain
in the ``hypothesis-sandbox`` direction and never promote directly to
supported/validated claims.

Categories
----------
- ``residue-class``    : statistical patterns in stopping times by residue class
- ``record-structure`` : structural properties of record-breaking seeds
- ``trajectory-shape`` : orbit shape classification and prediction
- ``algebraic-probe``  : searching for algebraic relationships
- ``orbit-structure``  : glide / odd-step fraction vs heuristics (Terras-type)
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metrics_sot import collatz_step, metrics_direct
from .repository import LabRepository, utc_now
from .schemas import ArtifactKind, Hypothesis, HypothesisStatus, Task, TaskStatus


DIRECTION_SLUG = "hypothesis-sandbox"

# Idempotency marker inside follow-up task descriptions (machine-readable).
_PROMISING_FOLLOWUP_MARKER = "[sandbox-promising-followup]"


def _sandbox_promising_followup_task_enabled() -> bool:
    """Env ``COLLATZ_SANDBOX_PROMISING_FOLLOWUP_TASK`` (default ``1``): auto-create checklist tasks."""
    v = os.getenv("COLLATZ_SANDBOX_PROMISING_FOLLOWUP_TASK", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def enqueue_sandbox_promising_followup_task(
    repository: LabRepository,
    *,
    claim_id: str,
    claim_title: str,
) -> Task | None:
    """Create a single open **review** task with a high-signal checklist, or return None if one exists.

    This automates the *nudge* from ``docs/HIGH_SIGNAL_EVIDENCE.md`` — not promotion,
    not proof, and not closing the loop without a human.
    """
    if not _sandbox_promising_followup_task_enabled():
        return None
    marker = f"{_PROMISING_FOLLOWUP_MARKER} claim_id={claim_id}"
    for t in repository.list_tasks():
        if t.status not in (TaskStatus.OPEN, TaskStatus.IN_PROGRESS):
            continue
        if marker in t.description or marker in t.title:
            return None

    description = f"""{marker}

Automated nudge: hypothesis-sandbox claim status is PROMISING (exploratory statistical probe).

Checklist — see docs/HIGH_SIGNAL_EVIDENCE.md:
1. Freeze context (git commit, range/end, kernel, env).
2. Re-run the same probe (API or CLI) on the same workspace.
3. Second SoT check on a tiny range (metrics_sot / collatz_step only).
4. If the signal is statistical: run battery stability across scales.
5. Add a research/ note: exact hypothesis, what would falsify it, numeric limits.

Claim id: {claim_id}
Probe title: {claim_title}
"""
    return repository.create_task(
        direction_slug=DIRECTION_SLUG,
        title=f"Review PROMISING sandbox probe {claim_id}",
        kind="review",
        description=description,
        owner="hypothesis-sandbox",
        priority=1,
    )


def _report_meta(
    *,
    probe_kind: str,
    range_start: int,
    range_end: int,
    sample_size: int | None = None,
    bin_spec: str | None = None,
    odd_stride: int | None = None,
    stability_endpoints: list[int] | None = None,
    bootstrap_reps: int | None = None,
    suggested_falsification: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Standard metadata block prepended or merged into probe evidence (English, machine-oriented)."""
    meta: dict[str, Any] = {
        "probe_kind": probe_kind,
        "range_start": range_start,
        "range_end": range_end,
        "sample_size": sample_size,
        "bin_spec": bin_spec,
        "odd_stride": odd_stride,
        "stability_endpoints": stability_endpoints,
        "bootstrap_reps": bootstrap_reps,
        "suggested_falsification": suggested_falsification or None,
        "extra": extra,
    }
    return {k: v for k, v in meta.items() if v is not None}


def _bootstrap_mean_ci(
    values: list[float],
    *,
    n_boot: int = 400,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap percentile CI for the sample mean. Returns (mean, ci_low, ci_high)."""
    if not values:
        return (0.0, 0.0, 0.0)
    if len(values) == 1:
        x = values[0]
        return (x, x, x)
    rng = random.Random(seed)
    n = len(values)
    boot_means: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot_means.append(statistics.mean(sample))
    boot_means.sort()
    lo_i = max(0, min(int((alpha / 2) * n_boot), n_boot - 1))
    hi_i = max(0, min(int((1 - alpha / 2) * n_boot) - 1, n_boot - 1))
    return (statistics.mean(values), boot_means[lo_i], boot_means[hi_i])


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

    class_rows = [
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
    approx_n = sum(s.count for s in stats)
    evidence = [
        {
            "report_meta": _report_meta(
                probe_kind="residue_class_global",
                range_start=start,
                range_end=end,
                sample_size=approx_n,
                bin_spec=f"mod {modulus}",
                suggested_falsification=(
                    "Run stratified residue analysis; extend range; if signal vanishes when "
                    "controlling log₂(n), treat global z as likely magnitude-confounded."
                ),
                extra={"odd_only": odd_only, "z_cutoff_grand": 1.5},
            ),
        },
        *class_rows,
    ]

    if high_classes:
        high_residues = [s.residue for s in high_classes]
        statement = (
            f"Exploratory (global pooling): residues ≡ {high_residues} (mod {modulus}) show "
            f"higher mean TST than the cross-class mean (z > 1.5σ vs class means) on "
            f"[{start}, {end}]. Not a proof; magnitude confounding is possible — use "
            f"stratified residue analysis and scale-stability before treating as structure."
        )
        status = HypothesisStatus.PLAUSIBLE
    elif low_classes:
        low_residues = [s.residue for s in low_classes]
        statement = (
            f"Exploratory (global pooling): residues ≡ {low_residues} (mod {modulus}) show "
            f"lower mean TST than the cross-class mean (z < -1.5σ) on [{start}, {end}]. "
            f"Requires stratification and falsification checks; not evidence for a theorem."
        )
        status = HypothesisStatus.PLAUSIBLE
    else:
        statement = (
            f"No residue class mod {modulus} exceeds the exploratory 1.5σ cutoff vs pooled "
            f"class means over [{start}, {end}]. Grand mean = {grand_mean:.1f}, σ = {grand_stddev:.2f}."
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
# Orbit helpers (hypothesis-only; does not extend NumberMetrics contract)
# ---------------------------------------------------------------------------

def _orbit_odd_even_counts(seed: int) -> tuple[int, int, int]:
    """Return ``(total_steps, odd_map_steps, even_halve_steps)`` until ``1``.

    *Odd* steps count maps ``n -> 3n+1`` applied from an odd current value;
    *even* steps count halving.  Uses :func:`~collatz_lab.metrics_sot.collatz_step`
    as the single-step SoT.
    """
    if seed < 1:
        raise ValueError("seed must be >= 1")
    current = seed
    total = odd_steps = even_steps = 0
    while current != 1:
        if current & 1 == 0:
            current = collatz_step(current)
            even_steps += 1
        else:
            current = collatz_step(current)
            odd_steps += 1
        total += 1
    return total, odd_steps, even_steps


def _odd_stride_range(
    start: int,
    end: int,
    *,
    odd_stride: int = 1,
) -> range:
    """Odd integers in ``[start, end]``, taking every ``odd_stride``-th odd."""
    if odd_stride < 1:
        raise ValueError("odd_stride must be >= 1")
    first = start if (start & 1) else start + 1
    step = 2 * odd_stride
    return range(first, end + 1, step)


# ---------------------------------------------------------------------------
# 1b. Stratified residue analysis (log₂ bins — reduces Simpson confounding)
# ---------------------------------------------------------------------------

def analyze_residue_classes_stratified(
    modulus: int,
    start: int = 1,
    end: int = 100_000,
    *,
    odd_only: bool = True,
    bin_count: int = 8,
    z_threshold: float = 2.0,
    min_bins_consistent: int = 3,
    min_class_per_bin: int = 12,
    odd_stride: int = 1,
) -> HypothesisResult:
    """Residue-class mean TST compared **within** logarithmic magnitude bins.

    Pooling all seeds when comparing residue classes mixes ``log₂ n`` with TST;
    this probe holds magnitude roughly constant per bin before comparing classes.
    """
    if modulus < 2:
        raise ValueError("Modulus must be at least 2.")
    if bin_count < 2:
        raise ValueError("bin_count must be at least 2.")

    log_min = math.log2(max(start, 1))
    log_max = math.log2(max(end, 1))
    if log_max <= log_min:
        log_max = log_min + 1e-9
    bin_width = (log_max - log_min) / bin_count

    bin_data: list[dict[int, list[int]]] = [
        {r: [] for r in range(modulus)} for _ in range(bin_count)
    ]

    if odd_only:
        r_iter = _odd_stride_range(start, end, odd_stride=odd_stride)
    else:
        r_iter = range(max(1, start), end + 1, odd_stride)
    for n in r_iter:
        m = metrics_direct(n)
        log_n = math.log2(n)
        bi = min(int((log_n - log_min) / bin_width), bin_count - 1)
        bin_data[bi][n % modulus].append(m.total_stopping_time)

    per_cell: list[dict] = []
    hi_bins: dict[int, int] = {r: 0 for r in range(modulus)}
    lo_bins: dict[int, int] = {r: 0 for r in range(modulus)}

    for bi in range(bin_count):
        flat: list[tuple[int, int]] = []
        for r in range(modulus):
            for t in bin_data[bi][r]:
                flat.append((r, t))
        if len(flat) < 30:
            continue
        m_all = statistics.mean(t for _, t in flat)
        s_all = statistics.stdev(t for _, t in flat) if len(flat) > 1 else 0.0
        if s_all <= 0:
            continue
        for r in range(modulus):
            ts = bin_data[bi][r]
            if len(ts) < min_class_per_bin:
                continue
            mr = statistics.mean(ts)
            se = s_all / math.sqrt(len(ts))
            z = (mr - m_all) / se if se > 0 else 0.0
            per_cell.append({
                "bin": bi,
                "residue": r,
                "count": len(ts),
                "mean_tst": round(mr, 3),
                "bin_mean_tst": round(m_all, 3),
                "z": round(z, 3),
            })
            if z > z_threshold:
                hi_bins[r] += 1
            elif z < -z_threshold:
                lo_bins[r] += 1

    high_residues = [r for r, c in hi_bins.items() if c >= min_bins_consistent]
    low_residues = [r for r, c in lo_bins.items() if c >= min_bins_consistent]

    n_seeds = sum(len(bin_data[bi][r]) for bi in range(bin_count) for r in range(modulus))
    evidence = [
        {
            "report_meta": _report_meta(
                probe_kind="residue_class_stratified_log2",
                range_start=start,
                range_end=end,
                sample_size=n_seeds,
                bin_spec=f"{bin_count} log2(n) bins, mod {modulus}",
                odd_stride=odd_stride,
                suggested_falsification=(
                    "Increase range; re-run battery stability; lower z_threshold or "
                    "min_bins_consistent; if hits disappear at larger scale, treat as unstable."
                ),
                extra={
                    "z_threshold": z_threshold,
                    "min_bins_consistent": min_bins_consistent,
                    "odd_only": odd_only,
                },
            ),
            "stratification": "log2_bins",
            "bin_count": bin_count,
            "z_threshold": z_threshold,
            "min_bins_consistent": min_bins_consistent,
            "odd_stride": odd_stride,
            "per_bin_residue_z": per_cell[:200],
            "high_hit_bins": {str(k): v for k, v in hi_bins.items() if v},
            "low_hit_bins": {str(k): v for k, v in lo_bins.items() if v},
        }
    ]

    if high_residues:
        statement = (
            f"Stratified (log₂ bins={bin_count}) mod {modulus}: residues ≡ {high_residues} show "
            f"higher mean TST than bin peers in ≥{min_bins_consistent} bins (z>{z_threshold}) "
            f"on [{start}, {end}] (odd_stride={odd_stride}). "
            f"Consistent-within-bin anomaly — not a proof; falsify via larger range / stability report."
        )
        status = HypothesisStatus.PLAUSIBLE
    elif low_residues:
        statement = (
            f"Stratified mod {modulus}: residues ≡ {low_residues} are faster than bin peers "
            f"in ≥{min_bins_consistent} bins on [{start}, {end}]. Exploratory only; check scale stability."
        )
        status = HypothesisStatus.PLAUSIBLE
    elif not per_cell:
        statement = (
            f"Insufficient per-bin class counts for stratified mod {modulus} "
            f"over [{start}, {end}] (try larger range or smaller modulus / bins)."
        )
        status = HypothesisStatus.PROPOSED
    else:
        statement = (
            f"No residue class mod {modulus} stays anomalous across ≥{min_bins_consistent} "
            f"log₂ bins on [{start}, {end}] (z_threshold={z_threshold}). "
            f"Interpretation: no consistent within-magnitude signal under this probe (Simpson confounding likely for raw mod tests)."
        )
        status = HypothesisStatus.FALSIFIED

    return HypothesisResult(
        title=f"Stratified residue TST (log₂ bins) mod {modulus}",
        statement=statement,
        category="residue-class",
        status=status,
        test_methodology=(
            f"Partitioned {'odd' if odd_only else 'all'} seeds in [{start}, {end}] into "
            f"{bin_count} bins on log₂(n); within each bin, z-scored class mean TST vs "
            f"bin-wide mean using SE ≈ σ_bin/√n_class."
        ),
        test_range=f"{start}-{end}",
        evidence=evidence,
        falsification=(
            "A single bin with large |z| is weak evidence; the test looks for "
            "**consistent** cross-bin anomalies.  Counter: reduce z_threshold or "
            "inspect per-bin counts for sparsity."
        ),
    )


# ---------------------------------------------------------------------------
# 1c. Glide / odd-step fraction (Terras-type heuristic)
# ---------------------------------------------------------------------------

def analyze_glide_structure(
    start: int = 1,
    end: int = 50_000,
    *,
    modulus: int = 8,
    sample_cap: int = 6_000,
    deviation_threshold: float = 0.02,
    bootstrap_reps: int = 400,
) -> HypothesisResult:
    """Empirical odd-step fraction vs Terras heuristic ``1/log₂(3)``.

    For large ``n``, random models predict the fraction of Collatz steps that
    are ``3n+1`` maps (from odd values) tends to ``1/log₂(3) ≈ 0.63093``.
    Systematic deviation by residue class would be structurally interesting
    (and could support or stress-test heuristics).
    """
    if modulus < 2:
        raise ValueError("modulus must be at least 2.")
    if sample_cap < 50:
        raise ValueError("sample_cap should be at least 50.")

    first_odd = start if (start & 1) else start + 1
    odd_count = max(0, (end - first_odd) // 2 + 1) if end >= first_odd else 0
    if odd_count == 0:
        return HypothesisResult(
            title="Glide / odd-step fraction analysis",
            statement="No odd seeds in range.",
            category="orbit-structure",
            status=HypothesisStatus.FALSIFIED,
            test_methodology="N/A",
            test_range=f"{start}-{end}",
            evidence=[],
        )

    step_j = max(1, (odd_count + sample_cap - 1) // sample_cap)
    sampled: list[int] = []
    j = 0
    while j < odd_count and len(sampled) < sample_cap:
        sampled.append(first_odd + 2 * j)
        j += step_j

    terras = 1.0 / math.log2(3.0)
    global_fracs: list[float] = []
    by_res: dict[int, list[float]] = {r: [] for r in range(modulus)}

    for n in sampled:
        total, odd_s, _even_s = _orbit_odd_even_counts(n)
        if total <= 0:
            continue
        f_odd = odd_s / total
        global_fracs.append(f_odd)
        by_res[n % modulus].append(f_odd)

    if not global_fracs:
        return HypothesisResult(
            title="Glide / odd-step fraction analysis",
            statement="No orbit data.",
            category="orbit-structure",
            status=HypothesisStatus.FALSIFIED,
            test_methodology="N/A",
            test_range=f"{start}-{end}",
            evidence=[],
        )

    mean_global, g_lo, g_hi = _bootstrap_mean_ci(global_fracs, n_boot=bootstrap_reps)
    delta_global = abs(mean_global - terras)
    pooled_sd = statistics.stdev(global_fracs) if len(global_fracs) > 1 else 0.0

    class_summary: list[dict] = []
    flagged: list[str] = []

    for r in range(modulus):
        xs = by_res[r]
        if len(xs) < 30:
            continue
        m, lo, hi = _bootstrap_mean_ci(xs, n_boot=bootstrap_reps)
        d = m - terras
        cohen_vs_sample = (m - mean_global) / pooled_sd if pooled_sd > 1e-12 else 0.0
        row = {
            "residue": r,
            "n": len(xs),
            "mean_odd_fraction": round(m, 5),
            "delta_vs_terras": round(d, 5),
            "effect_size_abs_vs_terras": round(abs(d), 5),
            "bootstrap_ci95_low": round(lo, 5),
            "bootstrap_ci95_high": round(hi, 5),
            "cohen_d_vs_global_sample": round(cohen_vs_sample, 4),
        }
        class_summary.append(row)
        ci_excludes = (terras < lo) or (terras > hi)
        if abs(d) > deviation_threshold and ci_excludes:
            flagged.append(
                f"≡{r} (mod {modulus}): mean={m:.4f}, Δ={d:+.4f}, 95% CI [{lo:.4f},{hi:.4f}] excludes Terras"
            )
        elif abs(d) > deviation_threshold:
            flagged.append(f"≡{r} (mod {modulus}): mean={m:.4f}, Δ={d:+.4f} (CI not excluding Terras)")

    evidence = [{
        "report_meta": _report_meta(
            probe_kind="glide_odd_fraction_terras",
            range_start=start,
            range_end=end,
            sample_size=len(global_fracs),
            bin_spec=f"mod {modulus} classes",
            bootstrap_reps=bootstrap_reps,
            suggested_falsification=(
                "Re-sample with larger sample_cap; run battery stability; if class deviations "
                "shrink or CIs cover Terras at larger scale, treat as sampling noise."
            ),
            extra={
                "deviation_threshold": deviation_threshold,
                "terras_reference": terras,
            },
        ),
        "terras_odd_fraction": round(terras, 6),
        "sampled_seeds": len(global_fracs),
        "global_mean_odd_fraction": round(mean_global, 6),
        "global_bootstrap_ci95_low": round(g_lo, 6),
        "global_bootstrap_ci95_high": round(g_hi, 6),
        "global_abs_delta_vs_terras": round(delta_global, 6),
        "global_pooled_sd_odd_fraction": round(pooled_sd, 6),
        "by_residue": class_summary,
        "flagged_classes": flagged,
    }]

    if flagged:
        statement = (
            f"Sample {len(sampled)} odd seeds in [{start}, {end}]: some mod-{modulus} classes "
            f"deviate from Terras baseline ({terras:.4f}) beyond threshold {deviation_threshold} "
            f"(with bootstrap 95% CIs on class means). Exploratory only — verify on larger "
            f"range and stability report; not evidence for global Collatz structure."
        )
        status = HypothesisStatus.PLAUSIBLE
    elif delta_global > deviation_threshold * 1.5 and (terras < g_lo or terras > g_hi):
        statement = (
            f"Global mean odd fraction {mean_global:.4f} (95% bootstrap CI [{g_lo:.4f}, {g_hi:.4f}]) "
            f"excludes Terras {terras:.4f} on this sample — warrants re-run with more seeds / "
            f"different range before interpreting."
        )
        status = HypothesisStatus.PROPOSED
    elif delta_global > deviation_threshold * 1.5:
        statement = (
            f"Global mean {mean_global:.4f} differs from Terras {terras:.4f} by {delta_global:.4f}, "
            f"but bootstrap CI still includes Terras — weak / inconclusive at this n."
        )
        status = HypothesisStatus.PROPOSED
    else:
        statement = (
            f"No strong deviation from Terras odd-step baseline on this sample: "
            f"global mean {mean_global:.4f} vs {terras:.4f}, n={len(sampled)}, "
            f"per-class checks within ±{deviation_threshold} (or CIs overlap Terras)."
        )
        status = HypothesisStatus.FALSIFIED

    return HypothesisResult(
        title=f"Glide structure (odd-step fraction) [{start}, {end}]",
        statement=statement,
        category="orbit-structure",
        status=status,
        test_methodology=(
            f"Uniform index-subsample of odd seeds up to {sample_cap}; "
            f"counted odd-map vs halving steps via SoT collatz_step to 1; "
            f"compared class means to 1/log₂(3); bootstrap CI on mean odd-fraction."
        ),
        test_range=f"{start}-{end}",
        evidence=evidence,
        falsification=(
            "Residue-specific drift does not prove the conjecture; it only constrains "
            "orbit statistics. A single threshold sweep can be noise — "
            "check stability across scales (battery scalability report)."
        ),
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
            f"Exploratory clustering only — not predictive evidence until tested on held-out ranges."
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
                f"Mean TST per log₂-bin tracks the Lagarias–Wagon heuristic "
                f"TST ≈ {LAGARIAS_CONSTANT:.2f} · log₂(n) on [{start}, {end}] "
                f"(mean |Δ%| ≈ {mean_deviation:.1f}%, max |Δ%| ≈ {max_deviation:.1f}%). "
                f"Descriptive fit only — not a proof."
            )
            status = HypothesisStatus.PLAUSIBLE
        elif max_deviation < 30:
            statement = (
                f"Approximate agreement with {LAGARIAS_CONSTANT:.2f} · log₂(n) on [{start}, {end}], "
                f"with moderate bin-level deviations (mean: {mean_deviation:.1f}%, max: {max_deviation:.1f}%). "
                f"May indicate correction terms or finite-range effects; requires falsification at larger n."
            )
            status = HypothesisStatus.PLAUSIBLE
        else:
            statement = (
                f"Heuristic TST ≈ {LAGARIAS_CONSTANT:.2f} · log₂(n) shows large bin-level gaps "
                f"(max |Δ%| ≈ {max_deviation:.1f}%) on [{start}, {end}]. "
                f"Treat as proposed model stress, not refutation of the conjecture."
            )
            status = HypothesisStatus.PROPOSED
    else:
        statement = "Insufficient data."
        status = HypothesisStatus.FALSIFIED

    approx_odd = max(0, (end - max(1, start)) // 2)
    evidence_out = [
        {
            "report_meta": _report_meta(
                probe_kind="stopping_time_growth_lagarias",
                range_start=start,
                range_end=end,
                sample_size=approx_odd,
                bin_spec=f"{bin_count} log2(n) bins",
                suggested_falsification=(
                    "Extend range; if max |Δ%| grows or shrinks systematically, revise bin count; "
                    "compare battery stability across ends."
                ),
                extra={"lagarias_constant": LAGARIAS_CONSTANT},
            ),
        },
        *evidence_bins,
    ]

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
        evidence=evidence_out,
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
            "report_meta": _report_meta(
                probe_kind="mod3_skip_redundancy_empirical",
                range_start=start,
                range_end=end,
                sample_size=tested,
                suggested_falsification=(
                    "Any single counterexample falsifies the skip rule in the tested range; "
                    "absence of counterexamples does not prove the global algebraic claim."
                ),
            ),
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
# Battery scalability — same probes at multiple ranges, flag status flips
# ---------------------------------------------------------------------------

def _scalability_odd_stride(end: int) -> int:
    """Keep work ~O(10⁵) odd seeds for very large ``end`` (explicit subsampling)."""
    approx_odds = max(1, end // 2)
    target = 120_000
    if approx_odds <= target:
        return 1
    return max(1, (approx_odds + target - 1) // target)


def run_battery_scalability_report(
    *,
    endpoints: list[int] | None = None,
    glide_sample_cap: int = 8_000,
    stratified_bin_count: int = 8,
    repository: LabRepository | None = None,
    persist: bool = False,
) -> dict[str, Any]:
    """Run a **fixed** probe set at each scale; list hypothesis statuses that change.

    Intended to falsify "artifacts of range size": if a pattern is real, it
    should be stable when the interval grows (subject to subsampling caps).

    When ``persist`` is True and ``repository`` is set, writes a JSON artifact
    under ``artifacts/hypotheses/`` (no new claim).
    """
    if endpoints is None:
        endpoints = [50_000, 200_000, 1_000_000]
    endpoints = sorted({int(e) for e in endpoints if int(e) >= 100})
    if len(endpoints) < 2:
        raise ValueError("Provide at least two distinct endpoints >= 100.")

    probe_slugs = (
        "stratified_mod8",
        "glide",
        "growth",
        "mod3_redundancy",
    )

    by_scale: dict[str, dict[str, str]] = {}
    raw_titles: dict[str, dict[str, str]] = {}

    for end in endpoints:
        stride = _scalability_odd_stride(end)
        res_s = analyze_residue_classes_stratified(
            8,
            start=1,
            end=end,
            bin_count=stratified_bin_count,
            odd_stride=stride,
        )
        res_g = analyze_glide_structure(
            1,
            end,
            sample_cap=glide_sample_cap,
        )
        res_gr = test_stopping_time_growth(1, end)
        res_m3 = test_mod3_convergence_redundancy(1, min(end, 10_000))

        by_scale[str(end)] = {
            "stratified_mod8": str(res_s.status),
            "glide": str(res_g.status),
            "growth": str(res_gr.status),
            "mod3_redundancy": str(res_m3.status),
        }
        raw_titles[str(end)] = {
            "stratified_mod8": res_s.title,
            "glide": res_g.title,
            "growth": res_gr.title,
            "mod3_redundancy": res_m3.title,
        }

    flips: list[dict[str, Any]] = []
    for slug in probe_slugs:
        seq = [by_scale[str(e)][slug] for e in endpoints]
        if len(set(seq)) > 1:
            for i in range(len(endpoints) - 1):
                a, b = seq[i], seq[i + 1]
                if a != b:
                    flips.append({
                        "probe": slug,
                        "from_end": endpoints[i],
                        "to_end": endpoints[i + 1],
                        "from_status": str(a),
                        "to_status": str(b),
                    })

    summary_lines: list[str] = []
    if not flips:
        summary_lines.append(
            "Scale-stability: no status flips between consecutive endpoints for the four tracked probes "
            f"({', '.join(probe_slugs)})."
        )
    else:
        summary_lines.append(
            f"Scale-sensitivity: {len(flips)} status transition(s) across scales — inspect probes below."
        )
        for f in flips:
            summary_lines.append(
                f"  • {f['probe']}: {f['from_status']} at end={f['from_end']} → "
                f"{f['to_status']} at end={f['to_end']}."
            )
    summary_lines.append(
        "Compare stratified_mod8 flips with odd_stride_by_scale (subsampling on large ranges)."
    )
    summary_lines.append(
        "mod3_redundancy uses a fixed cap min(end, 10_000); status often stable for large ends."
    )

    stability_verdict = "stable_across_scales" if not flips else "scale_sensitive_or_inconclusive"

    out: dict[str, Any] = {
        "report_meta": _report_meta(
            probe_kind="battery_scalability_meta",
            range_start=min(endpoints),
            range_end=max(endpoints),
            stability_endpoints=list(endpoints),
            suggested_falsification=(
                "If flips align with odd_stride jumps only, re-run with aligned subsampling; "
                "if a probe flips under fixed methodology, treat sandbox status as range-dependent."
            ),
            extra={"glide_sample_cap": glide_sample_cap, "stratified_bin_count": stratified_bin_count},
        ),
        "endpoints": endpoints,
        "odd_stride_by_scale": {str(e): _scalability_odd_stride(e) for e in endpoints},
        "by_scale": by_scale,
        "titles_by_scale": raw_titles,
        "status_flips": flips,
        "summary_lines": summary_lines,
        "stability_verdict": stability_verdict,
        "notes": (
            "Stratified residue analysis uses odd_stride>1 on huge ranges to bound work; "
            "interpret flips together with odd_stride_by_scale.  Mod-3 probe is always "
            "capped at min(end, 10_000) to match the main battery."
        ),
    }

    if persist and repository is not None:
        safe_ts = utc_now().replace(":", "-")
        rel_dir = repository.settings.artifacts_dir / "hypotheses"
        rel_dir.mkdir(parents=True, exist_ok=True)
        evidence_path = rel_dir / f"battery-stability-{safe_ts}.json"
        evidence_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        repository.create_artifact(
            kind=ArtifactKind.JSON,
            path=evidence_path,
            claim_id=None,
            metadata={
                "type": "battery-stability-report",
                "origin": "api-battery-stability",
            },
        )

    return out


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
        # No plain mod-8 pass: stratified mod-8 (log2 bins) is always run later in this battery.
        moduli = [3, 4, 6, 12, 16, 24]

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

    # 6. Stratified residue (log₂ bins) — mod 8 canonical probe
    results.append(
        analyze_residue_classes_stratified(
            8,
            start=1,
            end=end,
            bin_count=8,
            odd_stride=_scalability_odd_stride(end),
        )
    )

    # 7. Glide / odd-step fraction vs Terras heuristic
    results.append(analyze_glide_structure(1, end, sample_cap=8_000))

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
            "origin": "hypothesis-battery-api",
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
                "origin": "hypothesis-battery-api",
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


# Roadmap / CLI alias (same callable)
test_battery_scalability = run_battery_scalability_report
