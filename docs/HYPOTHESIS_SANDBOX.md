# Hypothesis sandbox — how it runs and how to tune it

## What exists

| Path | Role |
|------|------|
| **Worker** (`worker._try_hypothesis_experiment`) | One lightweight experiment every *N* completed verification runs, plus optionally during **idle** polling. Rotates (7-way): residue classes → record seeds → trajectory depths → growth rate → **mod‑3 structural redundancy** → **stratified residue (mod 8, log₂ bins)** → **glide / odd-step fraction**. |
| **`POST /api/hypotheses/battery`** | Full pack: all moduli + all generators + stratified mod 8 + glide in one call (creates several claims). For large `end`, stratified analysis uses the same `odd_stride` policy as the scalability report (~O(10⁵) odd seeds). |
| **`POST /api/hypotheses/battery/stability`** | The same structural probes at several values of `end` (default 50k → 200k → 1M); returns `status_flips` when a probe’s status changes between scales. |
| **Per-probe APIs** | `/api/hypotheses/residue-class`, `stratified-residue`, `glide-structure`, `record-seeds`, `trajectory-depths`, `growth-rate`. |
| **High-signal findings** | Escalation protocol after validated: `docs/HIGH_SIGNAL_EVIDENCE.md`. |
| **`cpu-barina` kernel** | Experimental **compute run** kernel (`compute_range_metrics` with `kernel=cpu-barina`). You can still queue it manually from **Operations**; the worker also rotates it via **kernel probes** (below). |

The `hypothesis.py` module emphasizes **falsification** and probes that can **refute** a pattern at larger scale, not only heuristic “support” for the conjecture.

## Why it felt “almost never”

1. **Old defaults:** hypothesis fired only after **20** CPU or **42** GPU verification runs — quiet labs rarely hit that.
2. **Restart reset:** `hypotheses_run` was memory-only, so every worker restart repeated the **same** first experiment (duplicate claims).
3. **Barina** is not a hypothesis cron job; it is a **kernel** you must select on a run.

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `COLLATZ_HYPOTHESIS_EVERY_N_RUNS` | `8` | After this many **completed** verification runs (CPU worker), run one sandbox experiment. |
| `COLLATZ_HYPOTHESIS_EVERY_N_RUNS_GPU` | `14` | Same for GPU-named workers (still CPU-side Python analysis). |
| `COLLATZ_HYPOTHESIS_IDLE_POLLS` | `30` | After this many **consecutive idle** polls (no queued run), run one experiment. Set `0` to disable. |
| `COLLATZ_KERNEL_PROBE_EVERY_N_RUNS` | `28` | After this many **completed** runs on this worker, enqueue one small `hypothesis-sandbox` run with the **next** kernel in rotation (`cpu-sieve`, `cpu-barina`, `cpu-parallel-odd`, … — filtered by what this worker supports). Owner `kernel-probe-worker`. Set `0` to disable. |
| `COLLATZ_HYPOTHESIS_ROTATION_MODE` | `full` | Sandbox auto-rotation: `full` = seven probes (incl. stratified + glide); `core` = five (no auto stratified/glide); `minimal` = residue + growth + mod‑3 only. Reduces DB noise when you want fewer exploratory claims. |
| `COLLATZ_SANDBOX_PROMISING_FOLLOWUP_TASK` | `1` | When `1`, each new **promising** sandbox claim triggers at most one **review** task (checklist → `docs/HIGH_SIGNAL_EVIDENCE.md`). Set `0` to disable. |

**Safety:** autopilot snacks, random probes, and kernel probes cap the **starting seed** for orbit-style kernels (`cpu-direct`, `cpu-accelerated`, `cpu-parallel`, `cpu-parallel-odd`, `gpu-collatz-accelerated`) so `range_end` stays ≤ **8_000_000** — otherwise a “small” span starting at 10¹⁰ could run effectively forever. Sieve kernels (`cpu-sieve`, `gpu-sieve`, `cpu-barina`) still use large random bases as before.

## Evidence shape (`report_meta`)

Structured probes attach a **`report_meta`** object (range, sample size, bin spec, optional `odd_stride`, suggested falsification text) so exports are comparable across probes. Battery / worker evidence JSON also sets **`origin`**: `hypothesis-battery-api` vs `hypothesis-worker-rotation` (and `rotation_mode` on worker artifacts).

## Battery stability persistence

`POST /api/hypotheses/battery/stability` accepts **`persist: true`** to write `artifacts/hypotheses/battery-stability-<timestamp>.json` and register an artifact (no claim). The response includes **`summary_lines`**, **`stability_verdict`**, and **`report_meta`** for the meta-probe.

## Persistent rotation

`data/hypothesis_sandbox_cursor_<worker_name>.json` stores `hypotheses_run` and `kernel_probe_index` so hypothesis + kernel rotation continue across restarts (legacy files with only `hypotheses_run` still load; `kernel_probe_index` defaults to `0`).

## Recommended workflows

- **Spike many hypotheses once:** Operations → Theory → **Run full hypothesis battery** (or `POST /api/hypotheses/battery` with `{ "end": 50000 }`).
- **Cross-scale robustness:** Operations → Theory → **Run battery stability report** (or `POST /api/hypotheses/battery/stability`) — same probes at 50k / 200k / 1M; UI shows flips + JSON download.
- **Barina path:** rely on **kernel probes** for occasional `cpu-barina` runs, or queue a specific interval manually; compare to `cpu-sieve` on the same interval if you need a controlled A/B check.
- **Tighter automation:** lower `COLLATZ_HYPOTHESIS_EVERY_N_RUNS*` or rely on idle polls on mostly-empty queues.
