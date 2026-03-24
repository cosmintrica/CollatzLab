# High-signal evidence — after “validated” and not losing results

This document explains what it means for a finding to be taken seriously in Collatz Lab and how we avoid losing a “wow” result in day-to-day noise.

## Levels (summary)

| Level | Role |
|--------|-----|
| **Compute + artifact** | Reproducible run, logs, JSON/metrics stored under `artifacts/`. |
| **Hypothesis sandbox** | Exploratory statuses (`proposed` / `plausible` / `falsified`) — *not* formal proofs. |
| **Validated (lab sense)** | Defined in the **correctness** protocol: SoT replay, kernel contract, CI differentials — see `docs/CORRECTNESS_AND_VALIDATION.md`. |
| **After validated (scientific escalation)** | Steps *outside* automation: formal claim, preprint/notes, independent verification, reproduction on another machine or codebase. |

**Important:** “Validated” in the lab means **engineering trust** in the implementation and reproducibility of numbers, **not** a proof of the Collatz conjecture.

## Automation (nudge only)

When a **hypothesis-sandbox** claim is stored with status **promising** (probe outcome `PLAUSIBLE`), the backend enqueues at most **one** open **review** task per claim id, carrying a shortened version of the checklist below. Duplicate tasks are suppressed while an open or in-progress task with the same claim marker exists. Disable with env `COLLATZ_SANDBOX_PROMISING_FOLLOWUP_TASK=0`.

This **does not** auto-promote a claim, does not mark anything as proven, and does not replace your judgment — it only makes “possible wow” visible on the **Tasks** board.

## Falsification, explicitly

- The **hypothesis sandbox** also targets **refuting** patterns (structural probes, cross-scale contradictions, `falsification` fields in payloads).
- The **`POST /api/hypotheses/battery/stability`** report runs the same probes at **50k → 200k → 1M** (or custom endpoints): a status that flips at larger scale suggests a **sampling / confounding artifact**, not “proof”.

## Checklist: something extraordinary — what do I do?

1. **Freeze context**  
   - Commit hash, backend version, `SEED`/interval, kernel, relevant environment variables.  
   - Copy the raw JSON artifact (not only a UI screenshot).

2. **Independent re-run**  
   - Run the same experiment from **CLI** or API on the same workspace.  
   - If you need speed, keep the same **SoT** (`metrics_direct` / `collatz_step`) for small checks.

3. **Second implementation (minimal)**  
   - For a single value or tiny interval: compare with an isolated Python script that only uses `metrics_sot`.  
   - Any mismatch → bug or misinterpretation, not a “theorem”.

4. **Dedicated claim + direction**  
   - Create a separate **claim** (not only a generic `hypothesis-sandbox` dump) with a title that states the *exact* hypothesis.  
   - Link **task/run** explicitly; avoid mixing with dozens of automated probes.

5. **Statistics / scale**  
   - For statistical patterns: run **battery stability** and record `odd_stride_by_scale` (subsampling on large intervals).  
   - If the signal disappears at a larger scale, treat it as **exploration**, not a stable result.

6. **Human review**  
   - Notes in `research/` with the hypothesis, what would **falsify** the result, and numeric limits.

7. **Promotion outside the sandbox**  
   - Only after the steps above: move toward a “serious” research track, optionally exportable material (generated report, PDF, public repo).

## What the lab does *not* do automatically

- It does not publish complete mathematical proofs.  
- It does not auto-promote from sandbox to “production truths” for conjecture verification.  
- It does not replace peer review or LaTeX formalization.

## Links

- Sandbox: `docs/HYPOTHESIS_SANDBOX.md`  
- Correctness / validated: `docs/CORRECTNESS_AND_VALIDATION.md`  
- Math probe roadmap: `research/ROADMAP.md` (section *Hypothesis Math Upgrades*)
