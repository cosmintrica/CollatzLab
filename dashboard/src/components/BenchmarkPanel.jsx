import { useCallback, useEffect, useMemo, useState } from "react";
import { apiBase, endpoints } from "../config.js";
import { readJson, readOptionalJson, postJson } from "../api.js";
import { StatusPill } from "./ui.jsx";
import { formatTimestamp } from "../utils.js";

const POLL_MS = 2500;
const MACHINE_KEY = "collatz_lab_machine";
const LAUNCH_KEY  = "collatz_bench_launch";

// ── Collatz orbit paths (module-level, computed once) ──────────────────────────
const _ORBIT_PATHS = (() => {
  function collatz(n) {
    const seq = [n];
    while (n !== 1 && seq.length < 350) {
      n = n % 2 === 0 ? n / 2 : 3 * n + 1;
      seq.push(n);
    }
    return seq;
  }
  function smoothPath(pts) {
    if (pts.length < 2) return "";
    let d = `M ${pts[0][0].toFixed(1)} ${pts[0][1].toFixed(1)}`;
    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = pts[Math.max(i - 1, 0)];
      const p1 = pts[i];
      const p2 = pts[i + 1];
      const p3 = pts[Math.min(i + 2, pts.length - 1)];
      const cp1x = p1[0] + (p2[0] - p0[0]) / 6;
      const cp1y = p1[1] + (p2[1] - p0[1]) / 6;
      const cp2x = p2[0] - (p3[0] - p1[0]) / 6;
      const cp2y = p2[1] - (p3[1] - p1[1]) / 6;
      d += ` C ${cp1x.toFixed(1)} ${cp1y.toFixed(1)}, ${cp2x.toFixed(1)} ${cp2y.toFixed(1)}, ${p2[0].toFixed(1)} ${p2[1].toFixed(1)}`;
    }
    return d;
  }
  const W = 1200, H = 56, pt = 5, pb = 8;
  return [27, 97, 703, 871, 6171].map(seed => {
    const seq = collatz(seed);
    const mx = Math.max(...seq);
    const step = W / Math.max(seq.length - 1, 1);
    const pts = seq.map((v, i) => [
      i * step,
      H - pb - (v / mx) * (H - pt - pb),
    ]);
    const peakI = seq.indexOf(mx);
    return { seed, d: smoothPath(pts), peak: pts[peakI] };
  });
})();

function clientLikelyDarwin() {
  if (typeof navigator === "undefined") return false;
  const p = (navigator.userAgentData?.platform || navigator.platform || "").toLowerCase();
  return p.includes("mac") || /Mac OS X/i.test(navigator.userAgent);
}

function useElapsed(active) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!active) { setElapsed(0); return; }
    const t0 = active.started_at ? new Date(active.started_at).getTime() : Date.now();
    const tick = () => setElapsed(Math.floor((Date.now() - t0) / 1000));
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [active]);
  return elapsed;
}

function fmtElapsed(s) {
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
}

function fmtThroughput(v) {
  const n = parseFloat(v);
  return isNaN(n) ? "—" : n.toFixed(2);
}

function fmtChunk(v) {
  const n = Number(v);
  if (!n) return "—";
  if (n >= 1e9) return `${(n / 1e9).toFixed(n % 1e9 === 0 ? 0 : 1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(n % 1e6 === 0 ? 0 : 1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`;
  return n.toLocaleString();
}

function fmtRunDuration(createdAt, finishedAt) {
  if (!createdAt || !finishedAt) return null;
  const secs = Math.round((new Date(finishedAt) - new Date(createdAt)) / 1000);
  if (secs <= 0) return null;
  if (secs < 60) return `${secs}s`;
  const m = Math.floor(secs / 60), s = secs % 60;
  return s > 0 ? `${m}m ${s}s` : `${m}m`;
}

function estimateSecs(linearEnd, reps, mps) {
  const t = mps > 0 ? mps : 400;
  return (linearEnd / 2) / (t * 1e6) * reps;
}

function fmtDur(secs) {
  if (!isFinite(secs) || secs <= 0) return null;
  if (secs < 2) return "<2s";
  if (secs < 90) return `~${Math.round(secs)}s`;
  return `~${(secs / 60).toFixed(1)} min`;
}

const PLATFORMS = [
  { id: "darwin",  label: "macOS",   live: true },
  { id: "linux",   label: "Linux",   live: false },
  { id: "windows", label: "Windows", live: false },
];

const BACKENDS = [
  { id: "gpu", label: "GPU · Metal",  live: true },
  { id: "cpu", label: "CPU · Native", live: false },
];

const DUR_TARGETS = [
  { label: "~30s",  t: 30 },
  { label: "~1 min", t: 60 },
  { label: "~2 min", t: 120 },
  { label: "~5 min", t: 300 },
];

const MEDAL = [
  null,
  { emoji: "🥇", label: "1st", cls: "bm-gold",   h: "tall" },
  { emoji: "🥈", label: "2nd", cls: "bm-silver", h: "mid" },
  { emoji: "🥉", label: "3rd", cls: "bm-bronze", h: "short" },
];

// ── Podium card ────────────────────────────────────────────────────────────────
function PodiumCard({ row, onDetail }) {
  const m = MEDAL[row.rank];
  if (!m) return null;
  return (
    <button
      type="button"
      className={`bm-podium-card bm-podium-${m.h} ${m.cls}`}
      onClick={() => onDetail(row.id)}
    >
      <span className="bm-podium-medal">{m.emoji}</span>
      <span className="bm-podium-rank">{m.label}</span>
      {row.machine_label && (
        <span className="bm-podium-machine">@{row.machine_label}</span>
      )}
      <span className="bm-podium-throughput">
        {fmtThroughput(row.throughput_m_per_s)}
        <span className="bm-podium-unit"> M odd/s</span>
      </span>
      <span className="bm-podium-chunk">chunk {fmtChunk(row.chunk_size)}</span>
      {row.parity_ok != null && (
        <span className={`bm-podium-parity ${row.parity_ok ? "ok" : "fail"}`}>
          {row.parity_ok ? "✓ parity ok" : "✗ parity fail"}
        </span>
      )}
      <span className="bm-podium-date">{formatTimestamp(row.finished_at)}</span>
    </button>
  );
}

// ── Running banner ─────────────────────────────────────────────────────────────
function RunningBanner({ active, elapsed }) {
  const meta = useMemo(() => {
    try { return JSON.parse(localStorage.getItem(LAUNCH_KEY) || "null"); } catch { return null; }
  }, []);
  const estSecs  = meta?.estSecs ?? 0;
  const pct      = estSecs > 0 ? Math.min(100, (elapsed / estSecs) * 100) : null;
  const overdue  = estSecs > 0 && elapsed > estSecs;
  const etaStr   = pct !== null
    ? overdue
      ? `+${fmtElapsed(elapsed - estSecs)} over estimate`
      : `~${fmtElapsed(Math.max(1, estSecs - elapsed))} left`
    : null;

  return (
    <div className="bm-running-banner" role="status" aria-live="polite">
      {/* Collatz orbit: n=27 (bright main) + n=6171 (ghost) */}
      <svg className="bm-orbit-svg" viewBox="0 0 1200 56" preserveAspectRatio="none" aria-hidden="true">
        <defs>
          <filter id="bm-orbit-glow" x="-40%" y="-200%" width="180%" height="500%">
            <feGaussianBlur in="SourceGraphic" stdDeviation="3.5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <linearGradient id="bm-shimmer-g" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="rgba(255,255,255,0)" />
            <stop offset="50%" stopColor="rgba(255,255,255,0.10)" />
            <stop offset="100%" stopColor="rgba(255,255,255,0)" />
          </linearGradient>
        </defs>
        {[14, 28, 42].map(y => (
          <line key={y} x1="0" y1={y} x2="1200" y2={y}
            stroke="rgba(255,255,255,0.04)" strokeWidth="0.5" />
        ))}
        {/* Ghost: seed 6171 (261 steps — dense chaotic texture) */}
        <path d={_ORBIT_PATHS[4].d} fill="none" className="bm-orbit-bg" strokeWidth="0.7" />
        {/* Main: seed 27 — famous 111-step sequence, spike to 9,232 */}
        <path d={_ORBIT_PATHS[0].d} fill="none" className="bm-orbit-main" strokeWidth="2.2"
          filter="url(#bm-orbit-glow)" />
        {/* Peak dot at 9,232 */}
        <circle cx={_ORBIT_PATHS[0].peak[0].toFixed(1)} cy={_ORBIT_PATHS[0].peak[1].toFixed(1)}
          r="3.5" className="bm-orbit-peak-main" />
        <rect className="bm-orbit-shimmer" x="-300" y="0" width="300" height="56"
          fill="url(#bm-shimmer-g)" />
      </svg>
      <div className="bm-running-body">
        <div className="bm-running-left">
          <span className="bm-running-dot" aria-hidden="true" />
          <div className="bm-running-label-stack">
            <span className="bm-running-label">Benchmark running</span>
            <span className="bm-running-sublabel">Collatz orbit · n=27 · peak 9,232 · 111 steps</span>
          </div>
          <code className="bm-running-jobid">{active.job_id?.slice(0, 8)}…</code>
        </div>
        <div className="bm-running-right">
          {etaStr && (
            <span className={`bm-running-eta${overdue ? " bm-running-eta--over" : ""}`}>{etaStr}</span>
          )}
          <div className="bm-running-elapsed-wrap">
            <span className="bm-running-elapsed-label">Elapsed</span>
            <span className="bm-running-elapsed">{fmtElapsed(elapsed)}</span>
          </div>
        </div>
      </div>
      {pct !== null && (
        <div className="bm-running-progress" aria-hidden="true">
          <div className={`bm-running-progress-fill${overdue ? " bm-running-progress-fill--over" : ""}`}
            style={{ width: `${pct}%` }} />
        </div>
      )}
    </div>
  );
}

// ── Detail panel ───────────────────────────────────────────────────────────────
function DetailPanel({ detail, loading }) {
  if (loading) return <p className="bm-muted">Loading…</p>;
  if (!detail) return <p className="bm-muted bm-detail-hint">Select a run to inspect it.</p>;
  const s = detail.summary ?? {};
  return (
    <div className="bm-detail">
      <div className="bm-detail-head">
        <code className="bm-detail-id">{detail.id}</code>
        <StatusPill value={detail.status ?? "unknown"} />
        {s.machine_label && (
          <span className="bm-detail-machine">@{s.machine_label}</span>
        )}
      </div>
      {s.winner_chunk != null && (
        <div className="bm-detail-hero">
          <div className="bm-detail-hero-item">
            <span className="bm-detail-hero-label">Winner chunk</span>
            <span className="bm-detail-hero-value">{fmtChunk(s.winner_chunk)}</span>
          </div>
          <div className="bm-detail-hero-item">
            <span className="bm-detail-hero-label">Throughput</span>
            <span className="bm-detail-hero-value">
              {fmtThroughput(s.winner_m_per_s)} <small>M odd/s</small>
            </span>
          </div>
          {s.winner_parity_ok != null && (
            <div className="bm-detail-hero-item">
              <span className="bm-detail-hero-label">Parity</span>
              <span className={`bm-detail-hero-value ${s.winner_parity_ok ? "ok" : "fail"}`}>
                {s.winner_parity_ok ? "✓ ok" : "✗ fail"}
              </span>
            </div>
          )}
          {fmtRunDuration(detail.created_at, detail.finished_at) && (
            <div className="bm-detail-hero-item">
              <span className="bm-detail-hero-label">Duration</span>
              <span className="bm-detail-hero-value">
                {fmtRunDuration(detail.created_at, detail.finished_at)}
              </span>
            </div>
          )}
        </div>
      )}
      {detail.error_message && <p className="bm-detail-err">{detail.error_message}</p>}
      <details className="bm-detail-raw">
        <summary>Raw JSON</summary>
        <pre className="bm-json-pre">{JSON.stringify(detail, null, 2)}</pre>
      </details>
    </div>
  );
}

// ── Main ───────────────────────────────────────────────────────────────────────
export default function BenchmarkPanel() {
  // Identity
  const [machineLabel, setMachineLabel] = useState(
    () => localStorage.getItem(MACHINE_KEY) || ""
  );
  const saveMachineLabel = (v) => {
    setMachineLabel(v);
    localStorage.setItem(MACHINE_KEY, v);
  };

  // View
  const [view, setView] = useState("hof");

  // HoF filters
  const [hofPlatform, setHofPlatform] = useState("darwin");
  const [hofBackend, setHofBackend] = useState("gpu");

  // Data
  const [status, setStatus] = useState(null);
  const [history, setHistory] = useState([]);
  const [hof, setHof] = useState([]);
  const [hofLoading, setHofLoading] = useState(false);
  const [detail, setDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState("");

  // Run form
  const [presets, setPresets] = useState([]);
  const [presetsLoading, setPresetsLoading] = useState(true);
  const [presetId, setPresetId] = useState("standard");
  const [quick, setQuick] = useState(false);
  const [linearEnd, setLinearEnd] = useState("");
  const [reps, setReps] = useState(5);
  const [warmup, setWarmup] = useState(2);
  const [chunksCsv, setChunksCsv] = useState("");
  const [writeCalibration, setWriteCalibration] = useState(true);
  const [pipelineAb, setPipelineAb] = useState(false);
  const [runPending, setRunPending] = useState(false);
  const [shutdownPending, setShutdownPending] = useState(false);

  const active = status?.active_job;
  const elapsed = useElapsed(active);
  const darwin = status?.darwin;
  const serverSystem = status?.server_system ?? "—";
  const metalOk = status?.metal_available;
  const runEnabled = Boolean(darwin && metalOk && !active);
  const apiHostMismatch = Boolean(clientLikelyDarwin() && !darwin);

  // Best throughput for duration estimates
  const bestMps = hof.length > 0 ? parseFloat(hof[0].throughput_m_per_s) || 400 : 400;

  // Duration estimate for custom form
  const linEndNum = parseInt(String(linearEnd), 10) || 0;
  const durEstStr = presetId === "custom" && linEndNum > 0
    ? fmtDur(estimateSecs(linEndNum, Number(reps) || 5, bestMps))
    : null;

  // ── Fetching ──────────────────────────────────────────────────────────────
  const refreshStatus = useCallback(async () => {
    try {
      const s = await readJson(endpoints.benchMetalChunkStatus);
      setStatus(s); setError("");
    } catch {
      setStatus(null);
      setError(`Cannot reach API at ${apiBase}. Start the backend.`);
    }
  }, []);

  const refreshHistory = useCallback(async () => {
    try {
      const rows = await readJson(endpoints.benchMetalChunkHistory(40));
      setHistory(Array.isArray(rows) ? rows : []);
    } catch { setHistory([]); }
  }, []);

  const refreshHof = useCallback(async () => {
    try {
      const rows = await readJson(endpoints.benchMetalChunkHallOfFame("Darwin", 50));
      setHof(Array.isArray(rows) ? rows : []);
    } catch { setHof([]); }
  }, []);

  useEffect(() => {
    refreshStatus();
    refreshHistory();
    setHofLoading(true);
    refreshHof().finally(() => setHofLoading(false));
  }, [refreshStatus, refreshHistory, refreshHof]);

  useEffect(() => {
    let cancelled = false;
    setPresetsLoading(true);
    readOptionalJson(endpoints.benchMetalChunkPresets)
      .then(d => { if (!cancelled) setPresets(Array.isArray(d) ? d : []); })
      .catch(() => { if (!cancelled) setPresets([]); })
      .finally(() => { if (!cancelled) setPresetsLoading(false); });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    const t = setInterval(() => {
      refreshStatus();
      refreshHistory();
      if (view === "hof") refreshHof();
    }, POLL_MS);
    return () => clearInterval(t);
  }, [refreshStatus, refreshHistory, refreshHof, view]);

  // ── Actions ───────────────────────────────────────────────────────────────
  const loadDetail = async (jobId) => {
    setDetailLoading(true);
    setView("history");
    try {
      setDetail(await readJson(endpoints.benchMetalChunkRunDetail(jobId)));
    } catch (e) {
      setDetail(null);
      setError(e?.message || "Could not load run detail.");
    } finally { setDetailLoading(false); }
  };

  const onStart = async (e) => {
    e.preventDefault();
    setRunPending(true);
    setError("");
    try {
      const payload = {
        preset: presetId,
        machine_label: machineLabel.trim() || undefined,
        quick,
        linear_end: linEndNum,
        reps: Math.max(1, Math.min(12, Number(reps) || 5)),
        warmup: Math.max(0, Math.min(5, Number(warmup) || 0)),
        chunks_csv: chunksCsv.trim(),
        write_calibration: writeCalibration,
        pipeline_ab: pipelineAb,
      };
      // Persist estimated duration for the progress bar
      const repsN = payload.reps;
      const estSecs = presetId === "custom" && linEndNum > 0
        ? Math.ceil(estimateSecs(linEndNum, repsN, bestMps))
        : null;
      localStorage.setItem(LAUNCH_KEY, JSON.stringify({ estSecs }));
      await postJson(endpoints.benchMetalChunkRun, payload, 60_000);
      await Promise.all([refreshStatus(), refreshHistory(), refreshHof()]);
      setView("history");
    } catch (err) {
      setError(err?.message || "Failed to start benchmark.");
    } finally { setRunPending(false); }
  };

  const releaseMetalHelper = async () => {
    setShutdownPending(true);
    try { await postJson(endpoints.gpuSieveMetalStdioShutdown, {}); }
    catch (err) { setError(err?.message || "Could not release Metal helper."); }
    finally { setShutdownPending(false); }
  };

  // Calculates linear_end for a target duration in seconds
  const applyDurPreset = (targetSecs) => {
    const repsN = Math.max(1, Number(reps) || 5);
    // linear_end = targetSecs * bestMps * 1e6 * 2 / repsN, rounded to nearest 100M
    const raw = targetSecs * bestMps * 1e6 * 2 / repsN;
    const rounded = Math.round(raw / 1e8) * 1e8;
    setLinearEnd(String(Math.max(rounded, 1e8)));
  };

  // ── Derived ───────────────────────────────────────────────────────────────
  const top3 = useMemo(() => hof.filter(r => r.rank <= 3), [hof]);
  const restHof = useMemo(() => hof.filter(r => r.rank > 3), [hof]);

  const presetCards = useMemo(() => {
    if (presets.length > 0) return presets;
    return [
      { id: "standard", title: "Standard", summary: presetsLoading ? "Loading…" : "Canonical 1M→16M chunk ladder · 5 reps · 2 warmup" },
      { id: "extended", title: "Extended", summary: presetsLoading ? "Loading…" : "Larger interval · same ladder · 5 reps · 2 warmup" },
      { id: "custom",   title: "Custom",   summary: "Full control over interval, chunks, reps" },
    ];
  }, [presets, presetsLoading]);

  const selectedPreset = presetCards.find(p => p.id === presetId);
  const showRealHof = hofPlatform === "darwin" && hofBackend === "gpu";

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <section className="tab-panel bm-page">

      {/* ── Header ── */}
      <header className="bm-header">
        {/* Top row: title + machine box */}
        <div className="bm-header-top">
          <div className="bm-header-text">
            <h1 className="bm-header-title">Benchmark</h1>
            <p className="bm-header-sub">
              Metal GPU chunk throughput — reproducible sweeps, scored on the API host.
            </p>
            <div className="bm-header-badges">
              <span className={`bm-badge ${darwin ? "bm-badge--ok" : "bm-badge--warn"}`}>
                <span className="bm-badge-dot" />
                {darwin ? "macOS API" : serverSystem}
              </span>
              <span className={`bm-badge ${!darwin ? "bm-badge--muted" : metalOk ? "bm-badge--ok" : "bm-badge--bad"}`}>
                <span className="bm-badge-dot" />
                {darwin ? (metalOk ? "Metal ready" : "Metal missing") : "Metal n/a"}
              </span>
            </div>
          </div>
          <div className="bm-machine-box">
            <label className="bm-machine-box-label" htmlFor="bm-machine-id">Lab identity</label>
            <input
              id="bm-machine-id"
              type="text"
              className="bm-machine-input"
              value={machineLabel}
              placeholder="e.g. M3 Pro · home-lab"
              maxLength={40}
              onChange={e => saveMachineLabel(e.target.value)}
            />
            <p className="bm-machine-hint">Shown next to your records in the Hall of Fame.</p>
          </div>
        </div>

        {/* Nav tabs — same design as filter bar, inside header */}
        <nav className="bm-header-nav" role="tablist" aria-label="Benchmark views">
          <div className="bm-filter-bar">
            {[
              { id: "hof",     label: "Hall of Fame", count: hof.length || null },
              { id: "run",     label: "Run",          count: null },
              { id: "history", label: "History",      count: history.length || null },
            ].map(tab => (
              <button
                key={tab.id} type="button" role="tab"
                aria-selected={view === tab.id}
                className={`bm-filter-btn bm-filter-btn--nav ${view === tab.id ? "bm-filter-btn--on" : ""}`}
                onClick={() => setView(tab.id)}
              >
                <span>{tab.label}</span>
                {tab.count != null && <span className="bm-filter-pill bm-filter-pill--count">{tab.count}</span>}
              </button>
            ))}
          </div>
        </nav>
      </header>

      {error && <div className="bm-error-banner">{error}</div>}
      {active && <RunningBanner active={active} elapsed={elapsed} />}

      {/* ════════════════ HOF ════════════════ */}
      {view === "hof" && (
        <div className="bm-hof">

          {/* Platform + backend filter row */}
          <div className="bm-filter-bar">
            <div className="bm-filter-group">
              {PLATFORMS.map(p => (
                <button
                  key={p.id} type="button"
                  className={`bm-filter-btn ${hofPlatform === p.id ? "bm-filter-btn--on" : ""} ${!p.live ? "bm-filter-btn--dim" : ""}`}
                  onClick={() => setHofPlatform(p.id)}
                >
                  {p.label}
                  {!p.live && <span className="bm-filter-pill">soon</span>}
                </button>
              ))}
            </div>
            <span className="bm-filter-divider" aria-hidden />
            <div className="bm-filter-group">
              {BACKENDS.map(b => (
                <button
                  key={b.id} type="button"
                  className={`bm-filter-btn ${hofBackend === b.id ? "bm-filter-btn--on" : ""} ${!b.live ? "bm-filter-btn--dim" : ""}`}
                  onClick={() => setHofBackend(b.id)}
                >
                  {b.label}
                  {!b.live && <span className="bm-filter-pill">soon</span>}
                </button>
              ))}
            </div>
          </div>

          {/* Coming soon */}
          {!showRealHof && (
            <div className="bm-hof-empty">
              <span className="bm-hof-empty-icon">🚧</span>
              <h2>Coming soon</h2>
              <p>
                {hofPlatform !== "darwin"
                  ? `${PLATFORMS.find(p => p.id === hofPlatform)?.label} benchmarks aren't tracked yet.`
                  : "CPU native sieve benchmarks aren't tracked yet."}
              </p>
            </div>
          )}

          {/* Real HoF */}
          {showRealHof && (
            <>
              {hofLoading && hof.length === 0 && (
                <p className="bm-muted bm-hof-loading">Loading…</p>
              )}
              {!hofLoading && hof.length === 0 && (
                <div className="bm-hof-empty">
                  <span className="bm-hof-empty-icon">🏆</span>
                  <h2>No entries yet</h2>
                  <p>Complete at least one Metal chunk sweep to appear here.</p>
                  <button type="button" className="bm-cta" onClick={() => setView("run")}>
                    Run first benchmark
                  </button>
                </div>
              )}
              {hof.length > 0 && (
                <>
                  <div className="bm-podium-stage">
                    {[
                      top3.find(r => r.rank === 2),
                      top3.find(r => r.rank === 1),
                      top3.find(r => r.rank === 3),
                    ].filter(Boolean).map(row => (
                      <PodiumCard key={row.id} row={row} onDetail={loadDetail} />
                    ))}
                  </div>
                  {restHof.length > 0 && (
                    <div className="bm-leaderboard">
                      <h3 className="bm-leaderboard-title">Full leaderboard</h3>
                      <div className="bm-lb-table-wrap">
                        <table className="bm-lb-table">
                          <thead>
                            <tr>
                              <th>#</th><th>Lab</th><th>M odd/s</th><th>Chunk</th>
                              <th>End N</th><th>Parity</th><th>Finished</th><th>Run</th>
                            </tr>
                          </thead>
                          <tbody>
                            {restHof.map(row => (
                              <tr key={row.id}>
                                <td className="bm-lb-rank">{row.rank}</td>
                                <td className="bm-lb-machine">{row.machine_label || <span className="bm-muted">—</span>}</td>
                                <td className="bm-lb-val"><strong>{fmtThroughput(row.throughput_m_per_s)}</strong></td>
                                <td>{fmtChunk(row.chunk_size)}</td>
                                <td>{row.linear_end != null ? row.linear_end.toLocaleString() : "—"}</td>
                                <td>
                                  {row.parity_ok == null ? "—" : (
                                    <span className={row.parity_ok ? "bm-ok" : "bm-fail"}>
                                      {row.parity_ok ? "✓" : "✗"}
                                    </span>
                                  )}
                                </td>
                                <td className="bm-nowrap">{formatTimestamp(row.finished_at)}</td>
                                <td>
                                  <button type="button" className="bm-link" onClick={() => loadDetail(row.id)}>
                                    {row.id?.slice(0, 8)}…
                                  </button>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </>
              )}
            </>
          )}
        </div>
      )}

      {/* ════════════════ RUN ════════════════ */}
      {view === "run" && (
        <div className="bm-run">
          {(apiHostMismatch || (darwin && !metalOk) || active) && (
            <div className="bm-run-status">
              {apiHostMismatch && (
                <div className="bm-warn-box">
                  Browser looks like macOS but API is on <strong>{serverSystem}</strong>. Set <code>VITE_API_BASE_URL</code>.
                </div>
              )}
              {darwin && !metalOk && (
                <div className="bm-warn-box">
                  Metal helper missing — build it first:<br />
                  <code>bash scripts/native_sieve_kit/metal/build_metal_sieve_chunk.sh</code>
                </div>
              )}
              {active && (
                <div className="bm-info-box">
                  Benchmark running ({active.job_id?.slice(0, 8)}…) — wait for it to finish.
                </div>
              )}
            </div>
          )}

          <form className="bm-run-form" onSubmit={onStart}>

            {/* ── Preset segmented control ── */}
            <div className="bm-section">
              <div className="bm-section-head">
                <h2 className="bm-section-title">Workload preset</h2>
                <p className="bm-section-sub">Standard and Extended use a locked chunk ladder — comparable across machines.</p>
              </div>

              <div className="bm-seg">
                {presetCards.map(p => (
                  <button
                    key={p.id} type="button"
                    className={`bm-seg-btn ${presetId === p.id ? "bm-seg-btn--on" : ""}`}
                    onClick={() => setPresetId(p.id)}
                  >
                    <span className="bm-seg-title">{p.title}</span>
                    {p.id !== "custom" && p.interval_linear_end && p.reps && (
                      <span className="bm-seg-dur">
                        {fmtDur(estimateSecs(p.interval_linear_end, p.reps, bestMps)) || "<1s"}
                      </span>
                    )}
                  </button>
                ))}
              </div>

              {/* Preset detail */}
              {presetId !== "custom" && selectedPreset && (
                <div className="bm-spec">
                  <span className="bm-spec-tag">Locked spec — reproducible</span>
                  <dl className="bm-spec-dl">
                    {selectedPreset.interval_linear_end != null && (
                      <><dt>Interval</dt><dd><code>[1, {selectedPreset.interval_linear_end.toLocaleString()}]</code></dd></>
                    )}
                    {Array.isArray(selectedPreset.chunk_odds) && (
                      <><dt>Chunk ladder</dt><dd><code>{selectedPreset.chunk_odds.join(", ")}</code></dd></>
                    )}
                    {selectedPreset.reps != null && (
                      <><dt>Reps / warmup</dt><dd>{selectedPreset.reps} timed / {selectedPreset.warmup ?? 1} warmup</dd></>
                    )}
                    {selectedPreset.interval_linear_end && selectedPreset.reps && (
                      <><dt>Est. duration</dt>
                        <dd>
                          {fmtDur(estimateSecs(selectedPreset.interval_linear_end, selectedPreset.reps, bestMps)) || "<1s"}
                          <span className="bm-spec-note"> at {fmtThroughput(bestMps)} M odd/s — use Custom for longer runs</span>
                        </dd>
                      </>
                    )}
                  </dl>
                </div>
              )}

              {/* Custom fields */}
              {presetId === "custom" && (
                <div className="bm-custom-wrap">
                  {/* Duration shortcuts */}
                  {/* Duration quick-set */}
                  <div className="bm-dur-row">
                    <span className="bm-dur-row-label">Target</span>
                    <div className="bm-dur-btns">
                      {DUR_TARGETS.map(d => (
                        <button key={d.t} type="button" className="bm-dur-btn" onClick={() => applyDurPreset(d.t)}>
                          {d.label}
                        </button>
                      ))}
                    </div>
                    <span className="bm-dur-note">
                      Sets Linear end for ~{fmtThroughput(bestMps)} M odd/s. Actual time varies by hardware.
                    </span>
                  </div>

                  {/* Fields grid */}
                  <div className="bm-custom-grid">
                    <label className="bm-field">
                      <span className="bm-field-label">Linear end</span>
                      <input type="number" min={0} value={linearEnd} placeholder="0"
                        onChange={ev => setLinearEnd(ev.target.value)} />
                      <span className="bm-field-desc">
                        Upper bound N of [1, N] — main lever for duration.
                        0 = use quick flag default.
                        {durEstStr && <strong> Est. {durEstStr}.</strong>}
                      </span>
                    </label>

                    <label className="bm-field">
                      <span className="bm-field-label">Timed reps</span>
                      <input type="number" min={1} max={12} value={reps}
                        onChange={ev => setReps(ev.target.value)} />
                      <span className="bm-field-desc">
                        Sweeps per chunk size. More = more stable score. Recommended: 5–10.
                      </span>
                    </label>

                    <label className="bm-field">
                      <span className="bm-field-label">Warmup runs</span>
                      <input type="number" min={0} max={5} value={warmup}
                        onChange={ev => setWarmup(ev.target.value)} />
                      <span className="bm-field-desc">
                        Untimed sweeps before measurement. Warms Metal caches. 2 is usually enough.
                      </span>
                    </label>

                    <label className="bm-field bm-field--full">
                      <span className="bm-field-label">Chunk odds CSV</span>
                      <input type="text" value={chunksCsv}
                        placeholder="1048576, 2097152, 4194304, 8388608, 16777216"
                        onChange={ev => setChunksCsv(ev.target.value)} />
                      <span className="bm-field-desc">
                        Chunk sizes to test, comma-separated. Each is one Metal GPU dispatch.
                        Empty = canonical 1M→16M ladder.
                      </span>
                    </label>
                  </div>

                  {/* Quick interval toggle */}
                  <div className="bm-options" style={{marginTop: 0}}>
                    <div className="bm-toggle-row">
                      <div className="bm-toggle-text">
                        <span className="bm-toggle-name">Quick interval</span>
                        <span className="bm-toggle-hint">
                          Uses [1, 12M] instead of [1, 48M] when linear_end = 0. Faster, but less stable readings.
                        </span>
                      </div>
                      <button type="button" role="switch" className="bm-toggle"
                        aria-checked={quick}
                        onClick={() => setQuick(v => !v)} />
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* ── Options ── */}
            <div className="bm-section">
              <h2 className="bm-section-title">Options</h2>
              <div className="bm-options">
                <div className="bm-toggle-row">
                  <div className="bm-toggle-text">
                    <span className="bm-toggle-name">Write calibration JSON</span>
                    <span className="bm-toggle-hint">
                      Saves the winning chunk size to disk — Metal gpu-sieve will auto-select it for future compute runs.
                    </span>
                  </div>
                  <button type="button" role="switch" className="bm-toggle"
                    aria-checked={writeCalibration}
                    onClick={() => setWriteCalibration(v => !v)} />
                </div>
                <div className="bm-toggle-row">
                  <div className="bm-toggle-text">
                    <span className="bm-toggle-name">Stdio pipeline A/B</span>
                    <span className="bm-toggle-hint">
                      Diagnostic: benchmarks two stdio transport modes at the winning chunk size. Adds a second pass.
                    </span>
                  </div>
                  <button type="button" role="switch" className="bm-toggle"
                    aria-checked={pipelineAb}
                    onClick={() => setPipelineAb(v => !v)} />
                </div>
              </div>
            </div>

            {/* ── Submit ── */}
            <div className="bm-submit-row">
              <button type="submit" className="bm-cta" disabled={runPending || !runEnabled}>
                {runPending ? "Starting…" : active ? "Job running…" : "Start sweep"}
              </button>
              <p className={`bm-submit-hint ${runEnabled ? "bm-submit-hint--ok" : ""}`}>
                {active
                  ? "Another benchmark is already running."
                  : !darwin
                  ? `API is on ${serverSystem} — Metal only runs on macOS.`
                  : !metalOk
                  ? "Build metal_sieve_chunk first (see warning above)."
                  : "Ready — GPU · Metal · runs on the API host."}
              </p>
              {darwin && (
                <button type="button" className="bm-secondary"
                  disabled={shutdownPending} onClick={releaseMetalHelper}>
                  {shutdownPending ? "…" : "Release Metal helper"}
                </button>
              )}
            </div>
          </form>
        </div>
      )}

      {/* ════════════════ HISTORY ════════════════ */}
      {view === "history" && (
        <div className="bm-history-layout">
          <div className="bm-history-list">
            <h2 className="bm-section-title">Recent runs</h2>
            {history.length === 0 && <p className="bm-muted">No runs yet.</p>}
            <ul className="bm-history-ul">
              {history.map(row => (
                <li key={row.id}>
                  <button type="button"
                    className={`bm-history-item ${detail?.id === row.id ? "bm-history-item--active" : ""}`}
                    onClick={() => loadDetail(row.id)}>
                    <span className="bm-history-top">
                      <code className="bm-history-id">{row.id.slice(0, 8)}…</code>
                      <StatusPill value={row.status} />
                    </span>
                    <span className="bm-history-meta">
                      {formatTimestamp(row.finished_at)}
                      {fmtRunDuration(row.created_at, row.finished_at) &&
                        ` · ${fmtRunDuration(row.created_at, row.finished_at)}`}
                      {row.summary?.winner_chunk != null
                        ? ` · ${fmtChunk(row.summary.winner_chunk)} · ${fmtThroughput(row.summary.winner_m_per_s)} M odd/s`
                        : ""}
                    </span>
                    {row.error_message && (
                      <span className="bm-history-err">{row.error_message}</span>
                    )}
                  </button>
                </li>
              ))}
            </ul>
          </div>
          <div className="bm-detail-panel">
            <h2 className="bm-section-title">Run detail</h2>
            <DetailPanel detail={detail} loading={detailLoading} />
          </div>
        </div>
      )}

    </section>
  );
}
