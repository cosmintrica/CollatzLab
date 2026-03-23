# LLM Continuous Learning — Architecture Plan

> Status: **planned** — foundation to be built incrementally
> Last updated: 2026-03-23

---

## Problem

The current Gemini autopilot is stateless. Every invocation starts from zero context. It does not remember previous runs, discoveries, decisions, or patterns. It cannot recognise when a result is unusual because it has no baseline to compare against. It cannot propose entries to the research paper because it has no persistent judgement layer.

---

## Goal

Build a memory and learning loop so the LLM autopilot:
1. Accumulates knowledge across sessions without manual prompting
2. Detects anomalies and extraordinary results automatically
3. Proposes `paper.json` Section 5 entries when a WOW threshold is crossed
4. Improves its task generation quality over time based on what actually worked

---

## Architecture — Four Layers

### Layer 1 — Observation Store (DB)

New table: `llm_observations`

```
id            TEXT  PRIMARY KEY
created_at    TEXT  NOT NULL
kind          TEXT  NOT NULL   -- 'run_completed', 'anomaly', 'record_broken', 'pattern', 'wow_candidate'
run_id        TEXT             -- linked run if applicable
kernel        TEXT
range_start   INTEGER
range_end     INTEGER
metric_key    TEXT             -- e.g. 'stopping_time_max', 'depth_record', 'parity_ratio'
metric_value  REAL
baseline      REAL             -- expected value at time of observation
deviation     REAL             -- (metric_value - baseline) / baseline, signed
summary       TEXT             -- 1-3 sentence human-readable note (LLM-generated)
wow_score     REAL             -- 0.0–1.0, 1.0 = extraordinary
promoted      INTEGER DEFAULT 0  -- 1 = added to paper.json Section 5
```

### Layer 2 — Baseline Tracker

After each verified run, compute rolling statistics:
- Average stopping time for seeds in range (mean, stddev)
- Maximum stopping time seen so far across all verified ranges
- Stopping time growth rate vs. seed magnitude (expected: O(log n))
- Parity ratio: fraction of odd steps in orbit
- Depth record: deepest orbit seen for seeds in range

These are stored in `runtime_settings` as JSON under key `llm_baselines`.

Trigger: `post_run_hook()` called by the worker after status → completed.

### Layer 3 — Anomaly Detector

After each run completes, the backend calls `detect_anomalies(run)`:

```python
def detect_anomalies(run: Run, baselines: dict) -> list[Observation]:
    observations = []

    # Check stopping time distribution
    if run.max_stopping_time > baselines["stopping_time_max"] * 1.05:
        observations.append(Observation(
            kind="record_broken",
            metric_key="stopping_time_max",
            metric_value=run.max_stopping_time,
            baseline=baselines["stopping_time_max"],
            deviation=...,
            wow_score=compute_wow_score(deviation, kind="record"),
        ))

    # Check depth record
    # Check parity anomalies
    # Check unexpected orbit collapses
    # ...

    return observations
```

`wow_score` formula:
- `deviation < 5%` → score 0.0
- `deviation 5–20%` → score 0.1–0.3
- `deviation 20–100%` → score 0.3–0.7
- `deviation > 100%` → score 0.7–1.0
- New absolute record → score automatically ≥ 0.8

### Layer 4 — WOW Promoter

If any observation has `wow_score ≥ 0.8`:
1. Send the observation + last 3 related runs to Gemini
2. Prompt: "This is an extraordinary result. Draft a research entry for Section 5 of the paper. Include: what was found, why it is surprising, the run_id for reproducibility, and a one-line LaTeX formula if applicable."
3. Write the draft to `research/wow_candidates/YYYY-MM-DD_<run_id>.json`
4. Human reviews and manually copies to `paper.json` `entries[]` — the LLM proposes, the human decides

---

## Implementation Phases

### Phase 1 — Observation Store (1–2 days)
- [ ] Add `llm_observations` table to schema
- [ ] `post_run_hook()` in services.py: collect basic metrics after each completed run
- [ ] API endpoint `GET /api/llm/observations` to read stored observations

### Phase 2 — Baseline Tracker (1 day)
- [ ] Compute rolling stats in `post_run_hook()`
- [ ] Store in `runtime_settings` as `llm_baselines`
- [ ] Dashboard widget: show current baselines and deviation from them

### Phase 3 — Anomaly Detector (2–3 days)
- [ ] `detect_anomalies()` function in new `llm_memory.py` module
- [ ] Integrate after run validation (selective validator passes → anomaly check runs)
- [ ] `wow_score` computation with configurable thresholds

### Phase 4 — WOW Promoter (2–3 days)
- [ ] Gemini prompt template for Section 5 draft
- [ ] `wow_candidates/` directory + file writer
- [ ] Dashboard notification when a new WOW candidate exists
- [ ] Human review UI: show candidate, Accept (copies to paper.json) or Reject

### Phase 5 — Context Injection (ongoing)
- [ ] Before every autopilot task generation, inject last 10 observations as context
- [ ] Let Gemini use observation history to prioritise what to investigate next
- [ ] Track which tasks generated by autopilot led to observations — close the feedback loop

---

## Context Injection Prompt Template (Phase 5)

```
You are the Collatz Attack Lab autopilot. You have persistent memory of prior runs.

RECENT OBSERVATIONS (last 10):
{observations_json}

CURRENT BASELINES:
{baselines_json}

ACTIVE TASKS:
{tasks_json}

Given the above, generate the next 3 research tasks. Prioritise anything that could
explain the anomalies in RECENT OBSERVATIONS. Do not repeat tasks that have already
been explored without new evidence.
```

---

## WOW Criteria (subject to revision)

A result qualifies for Section 5 of the research paper if it meets ALL of:
- [ ] `wow_score ≥ 0.8` from anomaly detector
- [ ] Result is reproducible from a single run_id
- [ ] Result cannot be explained by known algorithmic behaviour (e.g. not just a larger range)
- [ ] Human co-author explicitly approves after independent verification
- [ ] At minimum one cross-validation run on a different kernel confirms the finding

---

## Notes

- The LLM is advisory only. It cannot write to `paper.json` directly without human approval.
- Every WOW candidate is stored even if rejected, to build a history of false positives.
- Phase 1–2 can be built without any Gemini API calls — pure statistical monitoring.
- The memory store is append-only. Observations are never deleted, only superseded.
