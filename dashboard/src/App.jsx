import { startTransition, useDeferredValue, useEffect, useRef, useState } from "react";

const apiBase = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

const endpoints = {
  summary: `${apiBase}/api/summary`,
  directions: `${apiBase}/api/directions`,
  tasks: `${apiBase}/api/tasks`,
  runs: `${apiBase}/api/runs`,
  claims: `${apiBase}/api/claims`,
  claimRunLinks: `${apiBase}/api/claim-run-links`,
  linkClaimRun: `${apiBase}/api/claims/link-run`,
  sources: `${apiBase}/api/sources`,
  artifacts: `${apiBase}/api/artifacts`,
  artifactContent: (artifactId) => `${apiBase}/api/artifacts/${artifactId}/content`,
  artifactDownload: (artifactId) => `${apiBase}/api/artifacts/${artifactId}/download`,
  consensusBaseline: `${apiBase}/api/consensus-baseline`,
  redditFeed: `${apiBase}/api/external/reddit/collatz?limit=10`,
  fallacyTags: `${apiBase}/api/review/fallacy-tags`,
  modularProbe: `${apiBase}/api/review/probes/modular`,
  hardware: `${apiBase}/api/workers/capabilities`,
  workers: `${apiBase}/api/workers`
};

const tabs = [
  { id: "overview", label: "Start Here" },
  { id: "live-math", label: "Live Math" },
  { id: "directions", label: "Tracks" },
  { id: "evidence", label: "Evidence" },
  { id: "queue", label: "Operations" },
  { id: "guide", label: "Guide" }
];

const liveMathSections = [
  { id: "live-trace", label: "Trace" },
  { id: "live-ledger", label: "Ledger" },
  { id: "live-records", label: "Records" }
];

const defaultDirectionOptions = [
  { slug: "verification", title: "Verification" },
  { slug: "inverse-tree-parity", title: "Inverse Tree Parity" },
  { slug: "lemma-workspace", title: "Lemma Workspace" }
];

const sourceTypeOptions = [
  "peer_reviewed",
  "preprint",
  "self_published",
  "forum",
  "blog",
  "qa",
  "wiki",
  "media",
  "internal"
];

const sourceClaimTypeOptions = [
  "open_problem_consensus",
  "partial_result",
  "computational_verification",
  "proof_attempt",
  "heuristic",
  "discussion"
];

const sourceStatusOptions = ["intake", "under_review", "flagged", "supported", "refuted", "context"];

const mapVariantOptions = ["unspecified", "standard", "shortcut", "odd_only", "inverse_tree"];

const rubricFieldOptions = [
  { key: "peer_reviewed", label: "Peer reviewed" },
  { key: "acknowledged_errors", label: "Acknowledged errors" },
  { key: "defines_map_variant", label: "Defines map variant" },
  { key: "distinguishes_empirical_from_proof", label: "Separates proof from evidence" },
  { key: "proves_descent", label: "Proves descent" },
  { key: "proves_cycle_exclusion", label: "Proves cycle exclusion" },
  { key: "uses_statistical_argument", label: "Uses statistical argument" },
  { key: "validation_backed", label: "Validation backed" }
];

const defaultFallacyCatalog = [
  {
    tag: "empirical-not-proof",
    label: "Empirical is not proof",
    description: "Finite computation is evidence, not a universal theorem."
  },
  {
    tag: "almost-all-not-all",
    label: "Almost all is not all",
    description: "Density results do not settle every integer."
  },
  {
    tag: "circular-descent",
    label: "Circular descent",
    description: "The source assumes the same global descent it claims to prove."
  },
  {
    tag: "unchecked-generalization",
    label: "Unchecked generalization",
    description: "A local pattern is promoted to all n without a valid universal step."
  },
  {
    tag: "reverse-tree-gap",
    label: "Reverse tree gap",
    description: "The inverse-tree picture is not enough without a forward implication."
  },
  {
    tag: "publishing-does-not-imply-validity",
    label: "Publication is not validation",
    description: "Posting or publishing a manuscript does not make it correct."
  },
  {
    tag: "variant-confusion",
    label: "Variant confusion",
    description: "Standard, shortcut, odd-only, or inverse-tree variants are mixed together."
  },
  {
    tag: "proof-by-large-search",
    label: "Proof by large search",
    description: "A large verified interval is treated as if it solved the problem."
  },
  {
    tag: "statistical-leap",
    label: "Statistical leap",
    description: "Average-case language is used to conclude a deterministic theorem."
  }
];

const claimRunRelationOptions = ["supports", "tests", "refutes", "motivates", "depends_on"];

const directionGuide = {
  verification: {
    label: "Evidence track",
    role: "Runs CPU/GPU sweeps, compares kernels, and falsifies weak heuristics.",
    success: "Finds reproducible evidence or real search-space reduction.",
    caution: "This is not the proof track by itself."
  },
  "inverse-tree-parity": {
    label: "Structure track",
    role: "Explores odd predecessors, parity vectors, and modular filters.",
    success: "Finds structural constraints that survive wider testing.",
    caution: "Reverse-tree intuition still needs a forward implication."
  },
  "lemma-workspace": {
    label: "Proof track",
    role: "Tracks lemmas, dependencies, counterexamples, and source review.",
    success: "Promotes exact claims toward formalization with linked evidence.",
    caution: "Claims stay provisional until evidence and review agree."
  }
};

const evidenceGuide = [
  {
    kind: "validated-result",
    title: "Validated result",
    detail: "A run that passed an independent replay. This is high-trust computational evidence."
  },
  {
    kind: "claim",
    title: "Claim",
    detail: "A mathematical statement. It is theory, not evidence, until runs or artifacts support it."
  },
  {
    kind: "artifact",
    title: "Artifact",
    detail: "A saved file: JSON output, report, note, or proof draft you can preview or download."
  },
  {
    kind: "run",
    title: "Raw run",
    detail: "A compute record. Useful, but lower-trust than a validated result until replay succeeds."
  }
];

const initialState = {
  summary: null,
  directions: [],
  tasks: [],
  runs: [],
  claims: [],
  claimRunLinks: [],
  sources: [],
  artifacts: [],
  baseline: null,
  redditFeed: null,
  fallacyTags: [],
  hardware: [],
  workers: [],
  error: "",
  loading: true,
  lastUpdated: ""
};

async function readJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed for ${url}`);
  }
  return response.json();
}

async function readOptionalJson(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      return null;
    }
    return response.json();
  } catch {
    return null;
  }
}

async function postJson(url, payload) {
  const options = {
    method: "POST",
    headers: {}
  };
  if (payload !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(payload);
  }
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `Request failed for ${url}`;
    try {
      const body = await response.json();
      message = body.detail ?? body.message ?? message;
    } catch {
      // Ignore JSON parse failures and fall back to the generic message.
    }
    throw new Error(message);
  }
  return response.json();
}

function fileSafeLabel(value) {
  return String(value ?? "evidence")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "") || "evidence";
}

function downloadText(filename, content, mimeType = "text/plain;charset=utf-8") {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function downloadJson(filename, payload) {
  downloadText(filename, JSON.stringify(payload, null, 2), "application/json;charset=utf-8");
}

function triggerDownload(filename, content, type) {
  const blob = new Blob([content], { type });
  const objectUrl = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = objectUrl;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(objectUrl);
}

function exportJsonFile(filename, payload) {
  triggerDownload(filename, `${JSON.stringify(payload, null, 2)}\n`, "application/json");
}

function exportTextFile(filename, payload) {
  triggerDownload(filename, payload, "text/plain;charset=utf-8");
}

function asList(payload, keys) {
  if (!payload) {
    return [];
  }
  if (Array.isArray(payload)) {
    return payload;
  }
  for (const key of keys) {
    if (Array.isArray(payload[key])) {
      return payload[key];
    }
  }
  return [];
}

function prettyLabel(value) {
  return String(value).replaceAll("-", " ");
}

function normalize(value) {
  return String(value ?? "").trim().toLowerCase();
}

function timestampValue(value) {
  if (!value) {
    return 0;
  }
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function formatTimestamp(value) {
  if (!value) {
    return "No timestamp";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString();
}

function formatRelativeTime(value) {
  if (!value) {
    return "unknown time";
  }
  const timestamp = Date.parse(value);
  if (!Number.isFinite(timestamp)) {
    return value;
  }
  const diffSeconds = Math.round((Date.now() - timestamp) / 1000);
  const abs = Math.abs(diffSeconds);
  if (abs < 60) {
    return diffSeconds >= 0 ? "just now" : "in a moment";
  }
  if (abs < 3600) {
    const minutes = Math.round(abs / 60);
    return diffSeconds >= 0 ? `${minutes}m ago` : `in ${minutes}m`;
  }
  if (abs < 86400) {
    const hours = Math.round(abs / 3600);
    return diffSeconds >= 0 ? `${hours}h ago` : `in ${hours}h`;
  }
  const days = Math.round(abs / 86400);
  return diffSeconds >= 0 ? `${days}d ago` : `in ${days}d`;
}

function latestTimestamp(...values) {
  return [...values]
    .filter(Boolean)
    .sort((left, right) => timestampValue(right) - timestampValue(left))[0] || "";
}

function artifactLabel(path, fallback) {
  if (!path) {
    return fallback;
  }
  const pieces = String(path).split(/[\\/]/);
  return pieces[pieces.length - 1] || fallback;
}

function includesQuery(values, query) {
  const needle = normalize(query);
  if (!needle) {
    return true;
  }
  return values.some((value) => normalize(value).includes(needle));
}

function parseCsvList(value) {
  return String(value ?? "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function appendCsvTag(value, tag) {
  const next = parseCsvList(value);
  if (!next.includes(tag)) {
    next.push(tag);
  }
  return next.join(", ");
}

function rubricValueToSelect(value) {
  if (value === true) {
    return "yes";
  }
  if (value === false) {
    return "no";
  }
  return "unknown";
}

function selectToRubricValue(value) {
  if (value === "yes") {
    return true;
  }
  if (value === "no") {
    return false;
  }
  return null;
}

function countBy(items, getKey) {
  return items.reduce((accumulator, item) => {
    const key = getKey(item) || "unknown";
    accumulator[key] = (accumulator[key] || 0) + 1;
    return accumulator;
  }, {});
}

function StatusPill({ value }) {
  return <span className={`status-pill status-${value}`}>{prettyLabel(value)}</span>;
}

function SummaryCard({ label, value, note }) {
  return (
    <article className="summary-card">
      <span className="summary-label">{label}</span>
      <strong>{value}</strong>
      <p>{note}</p>
    </article>
  );
}

function SectionIntro({ title, text, action }) {
  return (
    <div className="section-intro">
      <div>
        <h2>{title}</h2>
        <p>{text}</p>
      </div>
      {action}
    </div>
  );
}

function formatMathNum(n) {
  const num = typeof n === "bigint" ? Number(n) : Number(n);
  if (num === 0) return { m: "0", e: null };
  const abs = Math.abs(num);
  if (abs < 100000) return { m: num.toLocaleString(), e: null };
  const e = Math.floor(Math.log10(abs));
  const mantissa = num / 10 ** e;
  const mStr = mantissa === Math.floor(mantissa) ? String(Math.floor(mantissa)) : mantissa.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
  return { m: mStr, e };
}

function MathNum({ value }) {
  const { m, e } = formatMathNum(value);
  if (e === null) return <>{m}</>;
  return <>{m}{"\u00b7"}10<sup>{e}</sup></>;
}

function buildTickerEquations(orbit) {
  const equations = [];
  if (orbit.length > 0) {
    for (let i = 0; i < orbit.length; i++) {
      const val = BigInt(orbit[i].value);
      if (val === 1n) {
        equations.push({ key: `t-${i}`, step: i, value: "1", nextValue: "1", isEven: true, terminal: true });
        break;
      }
      const isEven = val % 2n === 0n;
      const nextVal = isEven ? val / 2n : 3n * val + 1n;
      equations.push({ key: `t-${i}`, step: i, value: String(val), nextValue: String(nextVal), isEven, terminal: false });
    }
  }
  if (equations.length === 0) {
    return [
      { key: "idle-a", idle: true },
      { key: "idle-b", idle: true, alt: true },
      { key: "idle-c", idle: true },
      { key: "idle-d", idle: true, alt: true }
    ];
  }
  return equations;
}

function MathTicker({ run, orbit, frameIndex }) {
  const wrapRef = useRef(null);
  const trackRef = useRef(null);
  const segmentRef = useRef(null);
  const segmentWidthRef = useRef(0);
  const offsetRef = useRef(0);
  const animationRef = useRef(0);
  const lastTickRef = useRef(0);
  const pendingEquationsRef = useRef(null);
  const displayedSignatureRef = useRef("");
  const [segmentCopies, setSegmentCopies] = useState(4);
  const [displayedEquations, setDisplayedEquations] = useState(() => buildTickerEquations(orbit));
  const orbitSignature = orbit.length > 0 ? orbit.map((item) => item.value).join("|") : "idle";
  const speed = displayedEquations.length < 8 ? 22 : displayedEquations.length < 16 ? 32 : 42;

  useEffect(() => {
    const nextEquations = buildTickerEquations(orbit);
    if (displayedSignatureRef.current === "") {
      displayedSignatureRef.current = orbitSignature;
      setDisplayedEquations(nextEquations);
      return;
    }
    if (displayedSignatureRef.current === orbitSignature) {
      return;
    }
    pendingEquationsRef.current = {
      equations: nextEquations,
      signature: orbitSignature
    };
  }, [orbitSignature, orbit]);

  useEffect(() => {
    const segment = segmentRef.current;
    if (!segment) {
      return undefined;
    }

    const measure = () => {
      const previousWidth = segmentWidthRef.current;
      const width = segment.scrollWidth || segment.getBoundingClientRect().width || 0;
      const wrapWidth = wrapRef.current?.clientWidth || 0;
      if (previousWidth > 0 && width > 0) {
        const ratio = offsetRef.current / previousWidth;
        offsetRef.current = ratio * width;
      }
      segmentWidthRef.current = width;
      if (width > 0) {
        const neededCopies = Math.max(4, Math.ceil((wrapWidth || width) / width) + 3);
        setSegmentCopies((current) => (current === neededCopies ? current : neededCopies));
      }
      if (trackRef.current && width > 0) {
        if (offsetRef.current >= width) {
          offsetRef.current %= width;
        }
        trackRef.current.style.transform = `translate3d(${-offsetRef.current}px, 0, 0)`;
      }
    };

    measure();
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(measure);
    observer?.observe(segment);
    window.addEventListener("resize", measure);
    return () => {
      observer?.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [run?.id, displayedEquations]);

  useEffect(() => {
    const track = trackRef.current;
    if (!track) {
      return undefined;
    }

    let cancelled = false;
    const pixelsPerSecond = 96;
    lastTickRef.current = 0;

    const tick = (timestamp) => {
      if (cancelled) {
        return;
      }
      if (lastTickRef.current === 0) {
        lastTickRef.current = timestamp;
      }
      const deltaSeconds = (timestamp - lastTickRef.current) / 1000;
      lastTickRef.current = timestamp;
      const segmentWidth = segmentWidthRef.current;
      if (segmentWidth > 0) {
        offsetRef.current = (offsetRef.current + (deltaSeconds * pixelsPerSecond)) % segmentWidth;
        const pending = pendingEquationsRef.current;
        const seamDistance = Math.min(offsetRef.current, Math.abs(segmentWidth - offsetRef.current));
        if (pending && seamDistance < 6) {
          pendingEquationsRef.current = null;
          displayedSignatureRef.current = pending.signature;
          offsetRef.current = 0;
          setDisplayedEquations(pending.equations);
        }
        track.style.transform = `translate3d(${-offsetRef.current}px, 0, 0)`;
      }
      animationRef.current = window.requestAnimationFrame(tick);
    };

    animationRef.current = window.requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      if (animationRef.current) {
        window.cancelAnimationFrame(animationRef.current);
      }
    };
  }, [run?.id, speed]);

  return (
    <div className="math-ticker-wrap" aria-hidden="true" ref={wrapRef}>
      <svg className="ticker-staff" viewBox="0 0 100 72" preserveAspectRatio="none">
        <line x1="0" y1="1" x2="100" y2="1" />
        <line x1="0" y1="18" x2="100" y2="18" />
        <line x1="0" y1="36" x2="100" y2="36" />
        <line x1="0" y1="54" x2="100" y2="54" />
        <line x1="0" y1="71" x2="100" y2="71" />
      </svg>
      <div className="math-ticker-track" ref={trackRef}>
        {Array.from({ length: segmentCopies }).map((_, copyIndex) => (
          <div
            key={`segment-${copyIndex}`}
            className="math-ticker-segment"
            ref={copyIndex === 0 ? segmentRef : undefined}
          >
            {displayedEquations.map((eq) =>
              eq.idle ? (
                <span key={`${copyIndex}-${eq.key}`} className="ticker-eq">
                  <span className="ticker-formula">
                    {eq.alt ? <>T(n) = 3n + 1, &nbsp; n {"\u2261"} 1 (mod 2)</> : <>T(n) = n / 2, &nbsp; n {"\u2261"} 0 (mod 2)</>}
                  </span>
                </span>
              ) : eq.terminal ? (
                <span key={`${copyIndex}-${eq.key}`} className="ticker-eq">
                  <span className="ticker-formula">a<sub>{eq.step}</sub> {"\u2192"} 1 {"\u220e"}</span>
                  <span className="ticker-values">orbit converged</span>
                </span>
              ) : (
                <span key={`${copyIndex}-${eq.key}`} className="ticker-eq">
                  <span className="ticker-formula">
                    T(a<sub>{eq.step}</sub>) = {eq.isEven ? <>a<sub>{eq.step}</sub> / 2</> : <>3{"\u00b7"}a<sub>{eq.step}</sub> + 1</>}
                  </span>
                  <span className="ticker-values">
                    <MathNum value={eq.value} /> <span className="ticker-arrow">{"\u2192"}</span> <MathNum value={eq.nextValue} />
                  </span>
                </span>
              )
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function LiveMathNavigator({ runs, selectedRun, onSelectRun, onJumpToSection }) {
  const selectedRunIsLive = selectedRun?.status === "running";

  return (
    <div className="live-nav-bar">
      <div className="live-nav-left">
        <span className="orbit-kicker">Pinned</span>
        <strong>{selectedRun?.id ?? "none"}</strong>
        {selectedRun ? <span className="live-nav-detail">{selectedRun.kernel} | {selectedRun.hardware}</span> : null}
        {selectedRun ? (
          <span className={selectedRunIsLive ? "live-mode-badge live-mode-badge-live" : "live-mode-badge live-mode-badge-replay"}>
            {selectedRunIsLive ? "LIVE NOW" : "HISTORICAL REPLAY"}
          </span>
        ) : null}
      </div>
      <div className="live-nav-center">
        {liveMathSections.map((section) => (
          <button key={section.id} type="button" className="live-nav-button" onClick={() => onJumpToSection(section.id)}>
            {section.label}
          </button>
        ))}
      </div>
      <div className="live-nav-right">
        {runs.slice(0, 5).map((run) => (
          <button
            key={run.id}
            type="button"
            className={run.id === selectedRun?.id ? "live-nav-chip active" : "live-nav-chip"}
            onClick={() => onSelectRun(run.id)}
          >
            {run.id} {run.status === "running" ? "LIVE" : ""}
          </button>
        ))}
      </div>
    </div>
  );
}

function Legend() {
  const items = [
    ["active", "worth watching, but not proven strong yet"],
    ["promising", "has repeated support or real search-space reduction"],
    ["validated", "replayed successfully with independent logic"],
    ["refuted", "contradicted by evidence or failed reproduction"],
    ["frozen", "paused, retained for history, not deleted"]
  ];

  return (
    <div className="legend-grid">
      {items.map(([status, text]) => (
        <article key={status} className="legend-card">
          <StatusPill value={status} />
          <p>{text}</p>
        </article>
      ))}
    </div>
  );
}

function EmptyState({ title, text }) {
  return (
    <article className="empty-state">
      <h3>{title}</h3>
      <p>{text}</p>
    </article>
  );
}

function ShowMoreButton({ total, visible, onClick, label }) {
  if (total <= visible) {
    return null;
  }
  return (
    <button className="secondary-button" type="button" onClick={onClick}>
      Show more {label} ({total - visible} hidden)
    </button>
  );
}

function FilterField({ label, children }) {
  return (
    <label className="filter-field">
      <span>{label}</span>
      {children}
    </label>
  );
}

function FilterBar({ onClear, clearLabel = "Clear filters", children }) {
  return (
    <div className="filter-bar">
      {children}
      <button className="secondary-button" type="button" onClick={onClear}>
        {clearLabel}
      </button>
    </div>
  );
}

function CapabilityCard({ label, value, note }) {
  return (
    <article className="capability-card">
      <span>{label}</span>
      <strong>{value}</strong>
      <p>{note}</p>
    </article>
  );
}

function compactSourceLabel(url) {
  if (!url) {
    return "Source link unavailable";
  }
  try {
    const parsed = new URL(url);
    return parsed.hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

function baselineBadge(title) {
  const normalized = String(title || "").toLowerCase();
  if (normalized.includes("open problem")) {
    return "consensus";
  }
  if (normalized.includes("computational")) {
    return "evidence only";
  }
  if (normalized.includes("partial")) {
    return "partial result";
  }
  return "baseline";
}

function EvidenceDetailPanel({
  selectedKind,
  selectedRun,
  selectedClaim,
  selectedArtifact,
  relatedLinks,
  relatedRuns,
  relatedClaims,
  relatedArtifacts,
  previewPayload,
  onSelectEvidence,
  onPreviewArtifact,
  onExportJson,
  onExportText,
}) {
  const item = selectedRun || selectedClaim || selectedArtifact || null;
  const typeLabel = selectedRun
    ? selectedRun.status === "validated"
      ? "Validated result"
      : "Run"
    : selectedClaim
      ? "Claim"
      : selectedArtifact
        ? "Artifact"
        : "";
  const statusValue = selectedRun?.status || selectedClaim?.status || selectedArtifact?.kind || "";
  const showSecondaryStatus = !(
    selectedRun?.status === "validated" &&
    typeLabel === "Validated result"
  );
  const primaryArtifact = selectedArtifact || relatedArtifacts[0] || null;
  const validationMeaning = selectedRun?.status === "validated"
    ? `Validated result means this run was replayed through an independent implementation and the aggregate results matched. It does not mean a new Collatz formula was discovered; it means the compute evidence for this interval passed a stronger correctness check.`
    : "";

  if (!item) {
    return (
      <article className="panel evidence-detail-panel">
        <SectionIntro
          title="Evidence detail"
          text="Pick a validated result, claim, run, or artifact to inspect the full record and export it."
        />
        <EmptyState
          title="Nothing selected yet"
          text="The detail pane becomes active as soon as you open one evidence item from the lists below."
        />
      </article>
    );
  }

  return (
    <article className="panel evidence-detail-panel">
      <div className="card-head">
        <div>
          <p className="eyebrow">Evidence detail</p>
          <h3>{selectedClaim?.title || selectedRun?.name || selectedArtifact?.id}</h3>
          <p className="evidence-detail-subtitle">
            {typeLabel} | {item.id}
          </p>
        </div>
        <div className="evidence-detail-head">
          <span
            className={
              selectedRun?.status === "validated"
                ? "evidence-type-pill evidence-type-validated-result"
                : `evidence-type-pill evidence-type-${selectedKind || "run"}`
            }
          >
            {typeLabel}
          </span>
          {showSecondaryStatus ? <StatusPill value={statusValue} /> : null}
        </div>
      </div>

      <div className="detail-action-row">
        <button className="secondary-button" type="button" onClick={() => onExportJson(`${item.id}.json`, item)}>
          Export JSON
        </button>
        {primaryArtifact ? (
          <>
            <button className="secondary-button" type="button" onClick={() => onPreviewArtifact(primaryArtifact.id)}>
              Preview artifact
            </button>
            <a className="secondary-button detail-download-link" href={endpoints.artifactDownload(primaryArtifact.id)}>
              Download file
            </a>
          </>
        ) : null}
        {previewPayload ? (
          <button
            className="secondary-button"
            type="button"
            onClick={() => onExportText(`${previewPayload.artifact.id}.txt`, previewPayload.text)}
          >
            Export preview
          </button>
        ) : null}
      </div>

      {selectedRun ? (
        <>
          <p className="evidence-detail-lead">{selectedRun.summary || "No run summary stored yet."}</p>
          {selectedRun.status === "validated" ? (
            <div className="note-block">
              <p><strong>What this means</strong> {validationMeaning}</p>
              <p>
                <strong>For this record</strong> The interval {selectedRun.range_start} to {selectedRun.range_end} was checked,
                then replayed with an independent path, and the stored metrics matched.
              </p>
            </div>
          ) : null}
          <div className="metric-grid three-up">
            <div>
              <span className="metric-label">Direction</span>
              <strong>{selectedRun.direction_slug}</strong>
            </div>
            <div>
              <span className="metric-label">Interval</span>
              <strong>{selectedRun.range_start} to {selectedRun.range_end}</strong>
            </div>
            <div>
              <span className="metric-label">Kernel</span>
              <strong>{selectedRun.kernel}</strong>
            </div>
            <div>
              <span className="metric-label">Hardware</span>
              <strong>{selectedRun.hardware}</strong>
            </div>
            <div>
              <span className="metric-label">Processed</span>
              <strong>{selectedRun.metrics?.processed ?? selectedRun.metrics?.last_processed ?? "-"}</strong>
            </div>
            <div>
              <span className="metric-label">Checkpoint</span>
              <strong>{selectedRun.checkpoint?.last_processed ?? "-"}</strong>
            </div>
          </div>
          <div className="note-block">
            <p><strong>Max total stopping time</strong> {selectedRun.metrics?.max_total_stopping_time?.n ?? "-"} {"\u2192"} {selectedRun.metrics?.max_total_stopping_time?.value ?? "-"}</p>
            <p><strong>Max stopping time</strong> {selectedRun.metrics?.max_stopping_time?.n ?? "-"} {"\u2192"} {selectedRun.metrics?.max_stopping_time?.value ?? "-"}</p>
            <p><strong>Max excursion</strong> {selectedRun.metrics?.max_excursion?.n ?? "-"} {"\u2192"} {selectedRun.metrics?.max_excursion?.value ?? "-"}</p>
          </div>
        </>
      ) : null}

      {selectedClaim ? (
        <>
          <p className="evidence-detail-lead">{selectedClaim.statement}</p>
          <div className="metric-grid three-up">
            <div>
              <span className="metric-label">Direction</span>
              <strong>{selectedClaim.direction_slug}</strong>
            </div>
            <div>
              <span className="metric-label">Owner</span>
              <strong>{selectedClaim.owner}</strong>
            </div>
            <div>
              <span className="metric-label">Dependencies</span>
              <strong>{selectedClaim.dependencies?.length || 0}</strong>
            </div>
          </div>
          <div className="note-block">
            <p><strong>Dependencies</strong> {selectedClaim.dependencies?.length ? selectedClaim.dependencies.join(", ") : "none"}</p>
            <p><strong>Notes</strong> {selectedClaim.notes || "No notes stored yet."}</p>
          </div>
        </>
      ) : null}

      {selectedArtifact ? (
        <>
          <p className="evidence-detail-lead">{selectedArtifact.path}</p>
          <div className="metric-grid three-up">
            <div>
              <span className="metric-label">Kind</span>
              <strong>{selectedArtifact.kind}</strong>
            </div>
            <div>
              <span className="metric-label">Run</span>
              <strong>{selectedArtifact.run_id || "n/a"}</strong>
            </div>
            <div>
              <span className="metric-label">Claim</span>
              <strong>{selectedArtifact.claim_id || "n/a"}</strong>
            </div>
          </div>
          <p className="checksum">sha256: {selectedArtifact.checksum}</p>
        </>
      ) : null}

      <div className="detail-related-card detail-link-summary">
        <span className="metric-label">Claim to run links</span>
        {relatedLinks.length === 0 ? (
          <p>No explicit claim-to-run links exist for this record yet.</p>
        ) : (
          <div className="relation-list">
            {relatedLinks.map((link) => (
              <div key={`${link.claim_id}-${link.run_id}-${link.relation}`} className="relation-row">
                <strong>{link.claim_id}</strong>
                <span>{prettyLabel(link.relation)}</span>
                <strong>{link.run_id}</strong>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="detail-related-grid">
        <div className="detail-related-card">
          <span className="metric-label">Related runs</span>
          {relatedRuns.length === 0 ? (
            <p>No runs linked yet.</p>
          ) : (
            <div className="detail-link-list">
              {relatedRuns.map((run) => (
                <button
                  key={run.id}
                  className={selectedKind === "run" && selectedRun?.id === run.id ? "detail-link-button active" : "detail-link-button"}
                  type="button"
                  onClick={() => onSelectEvidence("run", run.id)}
                >
                  {run.id} | {run.status}
                </button>
              ))}
            </div>
          )}
        </div>
        <div className="detail-related-card">
          <span className="metric-label">Related claims</span>
          {relatedClaims.length === 0 ? (
            <p>No claims linked yet.</p>
          ) : (
            <div className="detail-link-list">
              {relatedClaims.map((claim) => (
                <button
                  key={claim.id}
                  className={selectedKind === "claim" && selectedClaim?.id === claim.id ? "detail-link-button active" : "detail-link-button"}
                  type="button"
                  onClick={() => onSelectEvidence("claim", claim.id)}
                >
                  {claim.id} | {claim.status}
                </button>
              ))}
            </div>
          )}
        </div>
        <div className="detail-related-card">
          <span className="metric-label">Related artifacts</span>
          {relatedArtifacts.length === 0 ? (
            <p>No artifacts linked yet.</p>
          ) : (
            <div className="detail-link-list">
              {relatedArtifacts.map((artifact) => (
                <button
                  key={artifact.id}
                  className={selectedKind === "artifact" && selectedArtifact?.id === artifact.id ? "detail-link-button active" : "detail-link-button"}
                  type="button"
                  onClick={() => onSelectEvidence("artifact", artifact.id)}
                >
                  {artifact.id} | {artifact.kind}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {previewPayload ? (
        <div className="detail-preview-card">
          <div className="card-head">
            <div>
              <span className="metric-label">Artifact preview</span>
              <strong>{previewPayload.artifact.id}</strong>
            </div>
            <button className="secondary-button" type="button" onClick={() => onExportText(`${previewPayload.artifact.id}.txt`, previewPayload.text)}>
              Export text
            </button>
          </div>
          <pre className="artifact-preview">{previewPayload.text.slice(0, 12000)}</pre>
          {previewPayload.text.length > 12000 ? <p className="meta-line">Preview truncated at 12,000 characters.</p> : null}
        </div>
      ) : null}
    </article>
  );
}

function ActionField({ label, wide = false, children }) {
  return (
    <label className={wide ? "action-field action-field-wide" : "action-field"}>
      <span>{label}</span>
      {children}
    </label>
  );
}

function firstPositiveInteger(...values) {
  for (const value of values) {
    const parsed = Number(value);
    if (Number.isFinite(parsed) && parsed >= 1) {
      return Math.floor(parsed);
    }
  }
  return 1;
}

function buildOrbit(seed, maxSteps = 18) {
  let current = typeof seed === "bigint" ? seed : BigInt(seed || 1);
  const frames = [];

  for (let step = 0; step < maxSteps; step += 1) {
    frames.push({
      step,
      value: current.toString()
    });
    if (current === 1n) {
      break;
    }
    current = current % 2n === 0n ? current / 2n : (3n * current) + 1n;
  }

  return frames;
}

function twoAdicValuation(value) {
  let current = BigInt(value || 0);
  let power = 0;
  while (current > 0n && current % 2n === 0n) {
    current /= 2n;
    power += 1;
  }
  return power;
}

function nextCollatzValue(value) {
  const current = BigInt(value || 1);
  if (current % 2n === 0n) {
    return {
      parity: "even",
      expression: `a_(k+1) = a_k / 2 = ${current} / 2 = ${current / 2n}`,
      next: (current / 2n).toString(),
      rule: `${current} = 0 (mod 2)`,
      acceleration: ""
    };
  }
  const next = (3n * current) + 1n;
  const valuation = twoAdicValuation(next);
  const compressed = next / (2n ** BigInt(valuation));
  return {
    parity: "odd",
    expression: `a_(k+1) = 3a_k + 1 = 3 * ${current} + 1 = ${next}`,
    next: next.toString(),
    rule: `${current} = 1 (mod 2)`,
    acceleration: `${next} = 2^${valuation} * ${compressed}`
  };
}

function orbitStats(orbit) {
  if (orbit.length === 0) {
    return { evenSteps: 0, oddSteps: 0, maxValue: "1" };
  }
  let evenSteps = 0;
  let oddSteps = 0;
  let maxValue = 0n;
  for (const item of orbit) {
    const value = BigInt(item.value);
    if (value % 2n === 0n) {
      evenSteps += 1;
    } else {
      oddSteps += 1;
    }
    if (value > maxValue) {
      maxValue = value;
    }
  }
  return {
    evenSteps,
    oddSteps,
    maxValue: maxValue.toString()
  };
}

function buildMathTrace(orbit, startIndex, kernel, limit = 5) {
  if (orbit.length === 0) {
    return [];
  }
  const rows = [];
  const firstIndex = Math.max(0, Math.min(startIndex, Math.max(0, orbit.length - 1)));

  for (let offset = 0; offset < limit; offset += 1) {
    const index = firstIndex + offset;
    if (index >= orbit.length) {
      break;
    }

    const current = BigInt(orbit[index].value);
    if (current === 1n) {
      rows.push({
        key: `${index}-terminal`,
        step: index,
        formula: `a_${index} = 1`,
        note: "The trace has entered the trivial loop 1 -> 4 -> 2 -> 1.",
        acceleration: ""
      });
      break;
    }

    if (current % 2n === 0n) {
      const next = current / 2n;
      rows.push({
        key: `${index}-even`,
        step: index,
        formula: `a_${index + 1} = a_${index} / 2 = ${current} / 2 = ${next}`,
        note: `${current} = 0 (mod 2), so the map contracts immediately.`,
        acceleration: ""
      });
      continue;
    }

    const lifted = (3n * current) + 1n;
    const valuation = twoAdicValuation(lifted);
    const compressed = lifted / (2n ** BigInt(valuation));
    rows.push({
      key: `${index}-odd`,
      step: index,
      formula: `a_${index + 1} = 3a_${index} + 1 = 3 * ${current} + 1 = ${lifted}`,
      note: `${current} = 1 (mod 2), so the direct trace takes the odd branch.`,
      acceleration:
        kernel === "cpu-accelerated"
          ? `${lifted} = 2^${valuation} * ${compressed}, so the accelerated odd kernel compresses to ${compressed}.`
          : `${lifted} = 2^${valuation} * ${compressed}.`
    });
  }

  return rows;
}

function summarizeMetric(metric) {
  if (metric === "max_total_stopping_time") {
    return {
      symbol: "sigma",
      label: "total stopping time",
      definition: "steps required to reach 1"
    };
  }
  if (metric === "max_stopping_time") {
    return {
      symbol: "tau",
      label: "descent time",
      definition: "first time the orbit falls below its seed"
    };
  }
  return {
    symbol: "E",
    label: "peak excursion",
    definition: "largest value reached on the orbit"
  };
}

function buildMetricSummary(run) {
  if (!run?.metrics) {
    return [];
  }
  return ["max_total_stopping_time", "max_stopping_time", "max_excursion"]
    .map((metric) => {
      const record = run.metrics?.[metric];
      if (!record || Number(record.n) < 1) {
        return null;
      }
      const details = summarizeMetric(metric);
      return {
        key: metric,
        label: details.label,
        formula: `${details.symbol}(${record.n}) = ${record.value}`,
        definition: details.definition
      };
    })
    .filter(Boolean);
}

function buildRecordTape(run) {
  const records = Array.isArray(run?.metrics?.sample_records) ? run.metrics.sample_records : [];
  return [...records]
    .reverse()
    .slice(0, 6)
    .map((record, index) => {
      const details = summarizeMetric(record.metric);
      return {
        key: `${record.metric}-${record.n}-${record.value}-${index}`,
        label: details.label,
        formula: `${details.symbol}(${record.n}) = ${record.value}`,
        definition: details.definition
      };
    });
}

function runProgress(run) {
  if (!run) {
    return { processed: 0, total: 0, percent: 0 };
  }
  const total = Math.max(0, Number(run.range_end) - Number(run.range_start) + 1);
  const processed = Math.max(0, Number(run.metrics?.processed || run.checkpoint?.last_processed || 0));
  const percent = total > 0 ? Math.min(100, (processed / total) * 100) : 0;
  return { processed, total, percent };
}

function describeCapability(capability) {
  if (!capability) {
    return "No capability record exposed yet.";
  }
  const executionReady = capability.metadata?.execution_ready;
  if (capability.kind === "gpu") {
    return executionReady
      ? `GPU path is executable with kernels: ${(capability.supported_kernels || []).join(", ")}.`
      : "GPU is detected but has no executable kernel yet, so the dashboard cannot dispatch real runs to it.";
  }
  return executionReady
    ? `CPU worker can execute: ${(capability.supported_kernels || []).join(", ")}.`
    : "CPU capability is visible but not executable yet.";
}

function orbitSeedFromRun(run) {
  return firstPositiveInteger(
    run?.checkpoint?.last_processed,
    run?.checkpoint?.next_value,
    run?.metrics?.max_total_stopping_time?.n,
    run?.metrics?.max_excursion?.n,
    run?.range_start
  );
}

function orbitSeedLabel(run) {
  if (!run) {
    return "no run selected";
  }
  if (run.status === "running" && Number(run.checkpoint?.last_processed) >= 1) {
    return `live checkpoint ${run.checkpoint.last_processed}`;
  }
  if (Number(run.metrics?.max_total_stopping_time?.n) >= 1) {
    return `record seed ${run.metrics.max_total_stopping_time.n}`;
  }
  return `seed ${run.range_start}`;
}

function describeOrbitSeed(run, orbitSeed) {
  const seed = BigInt(orbitSeed || 1);
  const valuation = twoAdicValuation(seed);
  const sourceLabel = run?.status === "running" && Number(run?.checkpoint?.last_processed) >= 1
    ? `checkpoint input n = ${run.checkpoint.last_processed}`
    : Number(run?.checkpoint?.last_processed) >= 1
      ? `saved checkpoint seed n = ${run.checkpoint.last_processed}`
      : Number(run?.metrics?.max_total_stopping_time?.n) >= 1
        ? `record seed n = ${run.metrics.max_total_stopping_time.n}`
        : `range seed n = ${run?.range_start ?? orbitSeed}`;
  const reconstructionNote = run?.status === "running"
    ? "The UI recomputes the next visible Collatz iterates from the latest real checkpoint input. This is exact for that seed, but it is not yet a direct inner-kernel trace dump."
    : "The UI is replaying the next visible Collatz iterates from a saved real seed. The formulas are exact for that seed, but they are reconstructed from stored run data.";
  const parityNote = valuation > 0
    ? `This seed is divisible by 2^${valuation}, so the first ${valuation} visible step${valuation === 1 ? "" : "s"} are immediate halvings. Similar leading blocks are expected when checkpoints land on nearby even inputs.`
    : "This seed is odd, so the next visible step should be a 3n + 1 expansion before any halving compression appears.";
  return {
    sourceLabel,
    detail: `${reconstructionNote} ${parityNote}`
  };
}

function OrbitPanel({ run, worker, frameIndex, onSelectRun, runs, expanded = false, showSelector = true, sectionId }) {
  const orbitSeed = run ? orbitSeedFromRun(run) : 1;
  const orbit = run ? buildOrbit(orbitSeed, 16) : [];
  const safeFrameIndex = orbit.length > 0 ? Math.min(frameIndex, orbit.length - 1) : 0;
  const frame = orbit.length > 0 ? orbit[safeFrameIndex] : null;
  const stepDetails = frame ? nextCollatzValue(frame.value) : null;
  const stats = orbitStats(orbit);
  const mathTrace = buildMathTrace(orbit, Math.max(0, safeFrameIndex - 1), run?.kernel, 5);
  const metricSummary = buildMetricSummary(run);
  const recordTape = buildRecordTape(run);
  const orbitSeedDetail = run ? describeOrbitSeed(run, orbitSeed) : null;

  return (
    <article
      id={sectionId}
      className={expanded ? "panel orbit-panel orbit-panel-expanded" : "panel orbit-panel"}
    >
      <SectionIntro
        title="Live mathematical trace"
        text="This panel takes a real seed from an actual run and rewrites the next Collatz iterates in mathematical language. When a worker is active, the seed comes from the newest checkpoint rather than a mock demo."
        action={
          <label className="orbit-run-picker">
            <span className="orbit-selector-label">Pinned run</span>
            <select
              value={run?.id ?? ""}
              onChange={(event) => onSelectRun(event.target.value)}
              aria-label="Select live math run"
            >
              {runs.length === 0 ? <option value="">No runs available</option> : null}
              {runs.slice(0, 8).map((item) => (
                <option key={item.id} value={item.id}>
                  {item.id} {item.status === "running" ? "- live now" : item.status ? `- ${item.status}` : ""}
                </option>
              ))}
            </select>
            <span className="orbit-picker-hint">
              {runs.some((item) => item.status === "running")
                ? "Showing active runs first while workers are live."
                : "No active runs right now. Showing the latest saved runs."}
            </span>
          </label>
        }
      />
      {run ? (
        <>
          <div className="orbit-header">
            <div>
              <span className="orbit-kicker">Selected run</span>
              <strong>{run.id}</strong>
              <p>{run.name}</p>
            </div>
            <div className="orbit-meta">
              <span>{run.kernel}</span>
              <span>{run.hardware}</span>
              <span>{run.status}</span>
            </div>
          </div>
          <div className="math-operator-banner">
            <span className="orbit-kicker">Collatz operator</span>
            <div className="math-expression">
              T(n) = n / 2 when n ≡ 0 (mod 2), &nbsp;&nbsp; T(n) = 3n + 1 when n ≡ 1 (mod 2)
            </div>
          </div>
          <div className="live-ledger">
            <article className="ledger-card">
              <span className="orbit-kicker">Current value</span>
              <strong>{frame?.value ?? orbitSeed}</strong>
              <p>{stepDetails?.rule ?? "Waiting for a frame."}</p>
            </article>
            <article className="ledger-card">
              <span className="orbit-kicker">Next direct image</span>
              <strong>{stepDetails?.next ?? orbitSeed}</strong>
              <p>{stepDetails?.expression ?? "No direct step available."}</p>
            </article>
            <article className="ledger-card">
              <span className="orbit-kicker">Worker checkpoint</span>
              <strong>
                {Number(run.checkpoint?.last_processed) >= 1
                  ? `n <= ${run.checkpoint.last_processed}`
                  : "No checkpoint"}
              </strong>
              <p>
                {worker
                  ? `${worker.name} | ${worker.status} | next ${run.checkpoint?.next_value ?? "unknown"}`
                  : `next ${run.checkpoint?.next_value ?? "unknown"} | waiting for worker trace`}
              </p>
            </article>
            <article className="ledger-card">
              <span className="orbit-kicker">Trace summary</span>
              <strong>{orbit.length} visible steps</strong>
              <p>{stats.oddSteps} odd, {stats.evenSteps} even, max value {stats.maxValue}</p>
            </article>
          </div>
          <div className="orbit-live-grid">
            <div className="orbit-visual-stack">
              <p className="orbit-note">
                {run.status === "running"
                  ? "The worker is writing checkpoints right now, so this trace uses the newest processed input n saved in the database."
                  : "This run is idle, so the trace replays a saved real seed: a checkpoint if available, otherwise a real record seed."}
              </p>
              {orbitSeedDetail ? (
                <article className="orbit-provenance-card">
                  <span className="orbit-kicker">Trace source</span>
                  <strong>{orbitSeedDetail.sourceLabel}</strong>
                  <p>{orbitSeedDetail.detail}</p>
                </article>
              ) : null}
              <div className="orbit-track" aria-label="Animated Collatz orbit">
                {orbit.map((item, index) => (
                  <div
                    key={`${run.id}-${item.step}`}
                    className={index === safeFrameIndex ? "orbit-square orbit-square-active" : "orbit-square"}
                  >
                    <span>a_{item.step}</span>
                    <strong>{item.value}</strong>
                  </div>
                ))}
              </div>
              <div className="orbit-footer">
                <div>
                  <span className="metric-label">Current frame</span>
                  <strong>{frame?.value ?? orbitSeed}</strong>
                </div>
                <div>
                  <span className="metric-label">{run.status === "running" ? "Checkpoint seed" : "Replay seed"}</span>
                  <strong>{orbitSeed}</strong>
                </div>
                <div>
                  <span className="metric-label">Processed values</span>
                  <strong>{run.metrics?.processed ?? 0}</strong>
                </div>
              </div>
            </div>
            <div className="orbit-equation-stack">
              <div className="equation-header">
                <span className="orbit-kicker">Equation tape</span>
                <p>Exact formulas recomputed from the selected real seed and frame. They are not placeholder data, but they are also not yet a direct kernel trace dump.</p>
              </div>
              <div className="equation-list">
                {mathTrace.map((row) => (
                  <article
                    key={row.key}
                    className={row.step === safeFrameIndex ? "equation-card equation-card-active" : "equation-card"}
                  >
                    <div className="equation-step-row">
                      <span>step {row.step}</span>
                      {row.step === safeFrameIndex ? <strong>live</strong> : null}
                    </div>
                    <div className="math-expression">{row.formula}</div>
                    <p>{row.note}</p>
                    {row.acceleration ? <p className="math-annotation">{row.acceleration}</p> : null}
                  </article>
                ))}
              </div>
            </div>
          </div>
          <div className="record-formula-grid">
            {metricSummary.map((record) => (
              <article key={record.key} className="record-formula-card">
                <span className="orbit-kicker">{record.label}</span>
                <div className="math-expression">{record.formula}</div>
                <p>{record.definition}</p>
              </article>
            ))}
          </div>
          <div className="record-tape">
            <div className="equation-header">
              <span className="orbit-kicker">Recorded events from this run</span>
              <p>These are the actual record updates saved in the run metrics.</p>
            </div>
            {recordTape.length === 0 ? (
              <p className="orbit-note">No sample record tape was stored for this run yet.</p>
            ) : (
              <div className="record-list">
                {recordTape.map((record) => (
                  <article key={record.key} className="record-row">
                    <div className="math-expression">{record.formula}</div>
                    <p>{record.label}: {record.definition}.</p>
                  </article>
                ))}
              </div>
            )}
          </div>
        </>
      ) : (
        <EmptyState
          title="No real run available yet"
          text="Start a run or queue one, then the live orbit will animate from an actual seed."
        />
      )}
      {showSelector ? (
        <div className="orbit-selector">
          <span className="orbit-selector-label">Switch to another real run</span>
          <div className="orbit-pills">
            {runs.slice(0, expanded ? 8 : 4).map((item) => (
              <button
                key={item.id}
                type="button"
                className={run?.id === item.id ? "orbit-pill active" : "orbit-pill"}
                onClick={() => onSelectRun(item.id)}
              >
                {item.id}
              </button>
            ))}
          </div>
        </div>
      ) : null}
    </article>
  );
}

function RunRail({ runs, selectedRunId, onSelectRun }) {
  return (
    <aside className="panel run-rail">
      <div className="run-rail-header">
        <span className="orbit-kicker">Run navigator</span>
        <span className="run-rail-count">{runs.length}</span>
      </div>
      {runs.length === 0 ? (
        <p className="orbit-note">No runs yet. Queue one first.</p>
      ) : (
        <div className="run-rail-list">
          {runs.slice(0, 10).map((run) => {
            const progress = runProgress(run);
            const selected = run.id === selectedRunId;
            return (
              <button
                key={run.id}
                type="button"
                className={selected ? "run-rail-card active" : "run-rail-card"}
                onClick={() => onSelectRun(run.id)}
              >
                <div className="run-rail-head">
                  <strong>{run.id}</strong>
                  <StatusPill value={run.status} />
                </div>
                <span className="run-rail-name">{run.name}</span>
                <span className="run-rail-meta">{run.kernel} | {run.hardware}</span>
                <div className="run-rail-progress">
                  <div className="run-rail-bar">
                    <div className="run-rail-fill" style={{ width: `${progress.percent}%` }} />
                  </div>
                  <span className="run-rail-pct">{progress.percent.toFixed(0)}%</span>
                </div>
              </button>
            );
          })}
        </div>
      )}
    </aside>
  );
}

function RedditIntelRail({ feed, onImportPost, pendingKey }) {
  const posts = Array.isArray(feed?.posts) ? feed.posts : [];
  const fetchedAt = feed?.fetched_at ? formatTimestamp(feed.fetched_at) : "not fetched yet";

  return (
    <aside className="workspace-rail">
      <article className="panel reddit-rail-card">
        <div className="card-head">
          <div>
            <p className="eyebrow">External watch</p>
            <h3>r/Collatz feed</h3>
          </div>
          <a
            className="secondary-button detail-download-link"
            href="https://www.reddit.com/r/Collatz/"
            target="_blank"
            rel="noreferrer"
          >
            Open Reddit
          </a>
        </div>
        <p className="reddit-rail-note">
          This is intake only. New posts can suggest sources to review, but nothing here is trusted automatically.
        </p>
        <div className="reddit-rail-stats">
          <article className="sidebar-runtime-card">
            <span>Fetched</span>
            <strong>{fetchedAt}</strong>
          </article>
          <article className="sidebar-runtime-card">
            <span>Review candidates</span>
            <strong>{feed?.review_candidate_count ?? 0}</strong>
          </article>
        </div>
        {posts.length === 0 ? (
          <EmptyState
            title="No subreddit feed yet"
            text="The backend has not returned the latest r/Collatz posts yet."
          />
        ) : (
          <div className="reddit-feed-list">
            {posts.map((post) => (
              <article key={post.id} className="reddit-post-card">
                <div className="card-head">
                  <span className={`reddit-signal-pill reddit-signal-${post.signal}`}>{prettyLabel(post.signal)}</span>
                  <span className="meta-line">{formatRelativeTime(post.created_at)}</span>
                </div>
                <strong>{post.title}</strong>
                <p>{post.excerpt}</p>
                <div className="reddit-post-meta">
                  <span>u/{post.author}</span>
                  <span>{post.score} score</span>
                  <span>{post.num_comments} comments</span>
                </div>
                <div className="card-action-row">
                  <a href={post.permalink} target="_blank" rel="noreferrer" className="secondary-button reddit-open-link">
                    Open thread
                  </a>
                  <button
                    type="button"
                    className="secondary-button reddit-open-link"
                    onClick={() => onImportPost(post)}
                    disabled={pendingKey === `reddit-${post.id}`}
                  >
                    {pendingKey === `reddit-${post.id}` ? "Importing..." : "Intake source"}
                  </button>
                </div>
              </article>
            ))}
          </div>
        )}
      </article>
    </aside>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState("overview");
  const [refreshNonce, setRefreshNonce] = useState(0);
  const [selectedEvidence, setSelectedEvidence] = useState({ kind: "", id: "" });
  const [artifactPreviews, setArtifactPreviews] = useState({});
  const [previewArtifactId, setPreviewArtifactId] = useState("");
  const [visible, setVisible] = useState({
    directions: 3,
    ledger: 8,
    runs: 6,
    claims: 4,
    sources: 4,
    artifacts: 4,
    tasks: 6
  });
  const [filters, setFilters] = useState({
    runQuery: "",
    runStatus: "all",
    runHardware: "all",
    taskQuery: "",
    taskStatus: "all",
    claimQuery: "",
    sourceQuery: "",
    sourceStatus: "all",
    artifactQuery: ""
  });
  const [quickForms, setQuickForms] = useState({
    runDirection: "verification",
    runName: "",
    runStart: "1",
    runEnd: "5000",
    runKernel: "cpu-parallel",
    runHardware: "auto",
    taskDirection: "verification",
    taskTitle: "",
    taskKind: "experiment",
    taskDescription: "",
    claimDirection: "lemma-workspace",
    claimTitle: "",
    claimStatement: "",
    linkClaimId: "",
    linkRunId: "",
    linkRelation: "supports",
    reviewDirection: "verification",
    sourceDirection: "lemma-workspace",
    sourceTitle: "",
    sourceAuthors: "",
    sourceYear: "2026",
    sourceUrl: "",
    sourceType: "self_published",
    sourceClaimType: "proof_attempt",
    sourceMapVariant: "unspecified",
    sourceSummary: "",
    sourceTags: "",
    sourceReviewId: "",
    sourceReviewStatus: "under_review",
    sourceReviewMapVariant: "unspecified",
    sourceReviewTags: "",
    sourceReviewNotes: "",
    sourceReviewPeerReviewed: "unknown",
    sourceReviewAcknowledgedErrors: "unknown",
    sourceReviewDefinesMapVariant: "unknown",
    sourceReviewDistinguishesProof: "unknown",
    sourceReviewProvesDescent: "unknown",
    sourceReviewProvesCycleExclusion: "unknown",
    sourceReviewUsesStatisticalArgument: "unknown",
    sourceReviewValidationBacked: "unknown",
    probeModulus: "8",
    probeResidues: "5",
    probeLimit: "255"
  });
  const [actionState, setActionState] = useState({
    pendingKey: "",
    tone: "",
    message: ""
  });
  const [reviewResult, setReviewResult] = useState(null);
  const [sourceReviewResult, setSourceReviewResult] = useState(null);
  const [probeResult, setProbeResult] = useState(null);
  const [navOpen, setNavOpen] = useState(false);
  const [orbitRunId, setOrbitRunId] = useState("");
  const [orbitFrame, setOrbitFrame] = useState(0);
  const [cpuPressure, setCpuPressure] = useState(null);
  const [state, setState] = useState(initialState);
  const deferredFilters = useDeferredValue(filters);
  const pollIntervalMs = state.runs.some((run) => run.status === "running") ? 3500 : 12000;

  useEffect(() => {
    let active = true;

    async function load() {
      try {
        const [summary, directions, tasks, runs, claims, claimRunLinks, sources, artifacts, baseline, redditFeed, fallacyTags, hardware, workers] = await Promise.all([
          readJson(endpoints.summary),
          readJson(endpoints.directions),
          readJson(endpoints.tasks),
          readJson(endpoints.runs),
          readJson(endpoints.claims),
          readOptionalJson(endpoints.claimRunLinks),
          readJson(endpoints.sources),
          readJson(endpoints.artifacts),
          readOptionalJson(endpoints.consensusBaseline),
          readOptionalJson(endpoints.redditFeed),
          readOptionalJson(endpoints.fallacyTags),
          readOptionalJson(endpoints.hardware),
          readOptionalJson(endpoints.workers)
        ]);
        if (!active) {
          return;
        }
        startTransition(() => {
          setState({
            summary,
            directions,
            tasks,
            runs,
            claims,
            claimRunLinks: asList(claimRunLinks, ["items", "links"]),
            sources,
            artifacts,
            baseline,
            redditFeed,
            fallacyTags: asList(fallacyTags, ["items", "tags", "catalog"]),
            hardware: asList(hardware, ["items", "hardware", "inventory"]),
            workers: asList(workers, ["items", "workers", "registry"]),
            error: "",
            loading: false,
            lastUpdated: new Date().toLocaleTimeString()
          });
        });
      } catch (error) {
        if (!active) {
          return;
        }
        startTransition(() => {
          setState((current) => ({
            ...current,
            error: error.message,
            loading: false
          }));
        });
      }
    }

    load();
    const timer = window.setInterval(load, pollIntervalMs);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [pollIntervalMs, refreshNonce]);

  useEffect(() => {
    if (typeof window.PressureObserver === "undefined") return;
    try {
      const observer = new window.PressureObserver((records) => {
        if (records.length > 0) {
          startTransition(() => setCpuPressure(records[records.length - 1]));
        }
      }, { sampleInterval: 1000 });
      observer.observe("cpu").catch(() => {});
      return () => observer.disconnect();
    } catch {
      // PressureObserver not available
    }
  }, []);

  const runHardwareCounts = countBy(state.runs, (run) => run.hardware);
  const runStatusCounts = countBy(state.runs, (run) => run.status);
  const cpuRuns = runHardwareCounts.cpu || 0;
  const gpuRuns = runHardwareCounts.gpu || 0;
  const mixedRuns = runHardwareCounts.mixed || 0;
  const queuedRuns = runStatusCounts.queued || 0;
  const runningRuns = runStatusCounts.running || 0;
  const validatedRunCount = runStatusCounts.validated || 0;
  const runningRunOptions = state.runs.filter((run) => run.status === "running");
  const activeCpuRuns = state.runs.filter((run) => run.status === "running" && run.hardware === "cpu").length;
  const activeGpuRuns = state.runs.filter((run) => run.status === "running" && run.hardware === "gpu").length;
  const activeMixedRuns = state.runs.filter((run) => run.status === "running" && run.hardware === "mixed").length;
  const hardwareInventory = state.hardware;
  const liveWorkers = state.workers.filter((worker) => {
    const status = normalize(worker.status || worker.state || worker.mode);
    return ["running", "active", "online", "idle", "busy"].includes(status) || status === "";
  });
  const visibleDirections = state.directions.slice(0, visible.directions);
  const filteredRuns = state.runs.filter((run) => {
    const query = deferredFilters.runQuery;
    const matchesStatus =
      deferredFilters.runStatus === "all" || run.status === deferredFilters.runStatus;
    const matchesHardware =
      deferredFilters.runHardware === "all" || run.hardware === deferredFilters.runHardware;
    return (
      matchesStatus &&
      matchesHardware &&
      includesQuery(
        [
          run.id,
          run.direction_slug,
          run.name,
          run.status,
          run.kernel,
          run.hardware,
          run.summary,
          run.range_start,
          run.range_end
        ],
        query
      )
    );
  });
  const filteredClaims = state.claims.filter((claim) =>
    includesQuery(
      [
        claim.id,
        claim.direction_slug,
        claim.title,
        claim.statement,
        claim.status,
        claim.owner,
        ...(claim.dependencies || [])
      ],
      deferredFilters.claimQuery
    )
  );
  const filteredSources = state.sources.filter((source) => {
    const matchesStatus =
      deferredFilters.sourceStatus === "all" || source.review_status === deferredFilters.sourceStatus;
    return (
      matchesStatus &&
      includesQuery(
        [
          source.id,
          source.direction_slug,
          source.title,
          source.authors,
          source.year,
          source.url,
          source.source_type,
          source.claim_type,
          source.review_status,
          source.map_variant,
          source.summary,
          source.notes,
          ...(source.fallacy_tags || [])
        ],
        deferredFilters.sourceQuery
      )
    );
  });
  const filteredArtifacts = state.artifacts.filter((artifact) =>
    includesQuery(
      [
        artifact.id,
        artifact.kind,
        artifact.path,
        artifact.checksum,
        artifact.run_id,
        artifact.claim_id
      ],
      deferredFilters.artifactQuery
    )
  );
  const filteredTasks = state.tasks.filter((task) => {
    const query = deferredFilters.taskQuery;
    const matchesStatus =
      deferredFilters.taskStatus === "all" || task.status === deferredFilters.taskStatus;
    return (
      matchesStatus &&
      includesQuery(
        [task.id, task.direction_slug, task.title, task.kind, task.description, task.owner],
        query
      )
    );
  });
  const visibleRuns = filteredRuns.slice(0, visible.runs);
  const visibleClaims = filteredClaims.slice(0, visible.claims);
  const visibleSources = filteredSources.slice(0, visible.sources);
  const visibleArtifacts = filteredArtifacts.slice(0, visible.artifacts);
  const visibleTasks = filteredTasks.slice(0, visible.tasks);
  const validatedRuns = filteredRuns.filter((run) => run.status === "validated");
  const fallacyCatalog = state.fallacyTags.length > 0 ? state.fallacyTags : defaultFallacyCatalog;
  const selectedReviewSource = state.sources.find((source) => source.id === quickForms.sourceReviewId) || null;
  const selectedOrbitRun = state.runs.find((run) => run.id === orbitRunId) || null;
  const selectedOrbitWorker =
    state.workers.find((worker) => worker.current_run_id === selectedOrbitRun?.id) ||
    state.workers.find((worker) => normalize(worker.status) === "running") ||
    null;
  const selectedRunProgress = runProgress(selectedOrbitRun);
  const cpuCapability =
    hardwareInventory.find((item) => normalize(item.kind) === "cpu") ||
    hardwareInventory.find((item) => normalize(item.slug).includes("cpu"));
  const gpuCapability =
    hardwareInventory.find((item) => normalize(item.kind) === "gpu") ||
    hardwareInventory.find((item) => normalize(item.slug).includes("gpu"));
  const selectedRunIsLive = selectedOrbitRun?.status === "running";
  const anyLiveRun = runningRuns > 0;
  const liveRunOptions = runningRunOptions.length > 0 ? runningRunOptions.slice(0, 8) : state.runs.slice(0, 8);
  const selectorRunOptions =
    selectedOrbitRun && !liveRunOptions.some((run) => run.id === selectedOrbitRun.id)
      ? [...liveRunOptions, selectedOrbitRun].slice(0, 8)
      : liveRunOptions;
  const runRailOptions = [
    ...runningRunOptions,
    ...state.runs.filter((run) => run.status !== "running")
  ].slice(0, 10);
  const cpuNowLabel =
    cpuCapability?.metadata?.usage_percent != null
      ? `${cpuCapability.metadata.usage_percent.toFixed(1)}%`
      : cpuPressure?.state
        ? prettyLabel(cpuPressure.state)
        : "n/a";
  const gpuNowLabel =
    gpuCapability?.metadata?.usage_percent != null
      ? `${gpuCapability.metadata.usage_percent.toFixed(1)}%`
      : gpuCapability
        ? "visible"
        : "n/a";
  const selectedEvidenceRun =
    selectedEvidence.kind === "run"
      ? state.runs.find((run) => run.id === selectedEvidence.id) || null
      : null;
  const selectedEvidenceClaim =
    selectedEvidence.kind === "claim"
      ? state.claims.find((claim) => claim.id === selectedEvidence.id) || null
      : null;
  const selectedEvidenceArtifact =
    selectedEvidence.kind === "artifact"
      ? state.artifacts.find((artifact) => artifact.id === selectedEvidence.id) || null
      : null;
  const selectedEvidenceLinks = state.claimRunLinks.filter((link) => {
    if (selectedEvidenceClaim) {
      return link.claim_id === selectedEvidenceClaim.id;
    }
    if (selectedEvidenceRun) {
      return link.run_id === selectedEvidenceRun.id;
    }
    if (selectedEvidenceArtifact?.claim_id) {
      return link.claim_id === selectedEvidenceArtifact.claim_id;
    }
    if (selectedEvidenceArtifact?.run_id) {
      return link.run_id === selectedEvidenceArtifact.run_id;
    }
    return false;
  });
  const relatedRuns = (() => {
    if (selectedEvidenceRun) {
      return [selectedEvidenceRun];
    }
    if (selectedEvidenceClaim) {
      const explicitRuns = selectedEvidenceLinks
        .map((link) => state.runs.find((run) => run.id === link.run_id))
        .filter(Boolean);
      const dependencyRuns = (selectedEvidenceClaim.dependencies || [])
        .map((runId) => state.runs.find((run) => run.id === runId))
        .filter(Boolean);
      return Array.from(new Map([...explicitRuns, ...dependencyRuns].map((run) => [run.id, run])).values());
    }
    if (selectedEvidenceArtifact?.run_id) {
      const run = state.runs.find((item) => item.id === selectedEvidenceArtifact.run_id);
      return run ? [run] : [];
    }
    return [];
  })();
  const relatedClaims = (() => {
    if (selectedEvidenceClaim) {
      return [selectedEvidenceClaim];
    }
    if (selectedEvidenceRun) {
      return selectedEvidenceLinks
        .map((link) => state.claims.find((claim) => claim.id === link.claim_id))
        .filter(Boolean);
    }
    if (selectedEvidenceArtifact?.claim_id) {
      const claim = state.claims.find((item) => item.id === selectedEvidenceArtifact.claim_id);
      return claim ? [claim] : [];
    }
    if (selectedEvidenceArtifact?.run_id) {
      return state.claimRunLinks
        .filter((link) => link.run_id === selectedEvidenceArtifact.run_id)
        .map((link) => state.claims.find((claim) => claim.id === link.claim_id))
        .filter(Boolean);
    }
    return [];
  })();
  const relatedArtifacts = state.artifacts.filter((artifact) => {
    if (selectedEvidenceArtifact) {
      return (
        artifact.id === selectedEvidenceArtifact.id ||
        (selectedEvidenceArtifact.run_id && artifact.run_id === selectedEvidenceArtifact.run_id) ||
        (selectedEvidenceArtifact.claim_id && artifact.claim_id === selectedEvidenceArtifact.claim_id)
      );
    }
    if (selectedEvidenceRun) {
      return artifact.run_id === selectedEvidenceRun.id;
    }
    if (selectedEvidenceClaim) {
      return artifact.claim_id === selectedEvidenceClaim.id;
    }
    return false;
  });
  const previewPayload = previewArtifactId ? artifactPreviews[previewArtifactId] || null : null;
  const evidenceLedger = [
    ...state.runs.map((run) => ({
      key: `run-${run.id}`,
      kind: run.status === "validated" ? "validated-result" : "run",
      id: run.id,
      title: run.name,
      status: run.status,
      direction: run.direction_slug,
      summary: run.summary || `${run.range_start} to ${run.range_end}`,
      timestamp: latestTimestamp(run.finished_at, run.started_at, run.created_at),
      meta: `${run.kernel} | ${run.hardware}`,
      openKind: "run"
    })),
    ...state.claims.map((claim) => ({
      key: `claim-${claim.id}`,
      kind: "claim",
      id: claim.id,
      title: claim.title,
      status: claim.status,
      direction: claim.direction_slug,
      summary: claim.statement,
      timestamp: latestTimestamp(claim.updated_at, claim.created_at),
      meta: claim.dependencies?.length ? `depends on ${claim.dependencies.join(", ")}` : "no dependencies",
      openKind: "claim"
    })),
    ...state.artifacts.map((artifact) => ({
      key: `artifact-${artifact.id}`,
      kind: "artifact",
      id: artifact.id,
      title: artifactLabel(artifact.path, artifact.id),
      status: artifact.kind,
      direction: artifact.claim_id || artifact.run_id || "artifact",
      summary: artifact.path,
      timestamp: latestTimestamp(artifact.created_at),
      meta: artifact.run_id ? `run ${artifact.run_id}` : artifact.claim_id ? `claim ${artifact.claim_id}` : "unlinked file",
      openKind: "artifact"
    }))
  ].sort((left, right) => timestampValue(right.timestamp) - timestampValue(left.timestamp));
  const visibleEvidenceLedger = evidenceLedger.slice(0, visible.ledger);
  const orbitFrames = selectedOrbitRun ? buildOrbit(orbitSeedFromRun(selectedOrbitRun), 16) : [];
  const supportedClaims = state.claims.filter((claim) =>
    ["supported", "promising", "formalize"].includes(claim.status)
  ).length;
  const flaggedSources = state.sources.filter((source) => source.review_status === "flagged").length;
  const reviewedSources = state.sources.filter((source) => source.review_status !== "intake").length;
  const hasLoaded = !state.loading && state.summary !== null;
  const summary = state.summary;
  const activeTabLabel = tabs.find((tab) => tab.id === activeTab)?.label ?? "Dashboard";
  const directionOptions = state.directions.length > 0 ? state.directions : defaultDirectionOptions;
  const kernelOptions = Array.from(
    new Set([
      "cpu-direct",
      "cpu-accelerated",
      "cpu-parallel",
      ...hardwareInventory.flatMap((item) => item.supported_kernels || [])
    ])
  );
  const hardwareOptions = Array.from(
    new Set([
      "cpu",
      "auto",
      ...hardwareInventory.flatMap((item) => item.supported_hardware || [])
    ])
  );
  const hasEvidence =
    hasLoaded &&
    (state.runs.length > 0 ||
      state.claims.length > 0 ||
      state.sources.length > 0 ||
      state.tasks.length > 0 ||
      state.artifacts.length > 0);

  function reviewFormPatch(source) {
    return {
      sourceReviewId: source?.id || "",
      sourceReviewStatus: source?.review_status || "under_review",
      sourceReviewMapVariant: source?.map_variant || "unspecified",
      sourceReviewTags: (source?.fallacy_tags || []).join(", "),
      sourceReviewNotes: source?.notes || "",
      sourceReviewPeerReviewed: rubricValueToSelect(source?.rubric?.peer_reviewed),
      sourceReviewAcknowledgedErrors: rubricValueToSelect(source?.rubric?.acknowledged_errors),
      sourceReviewDefinesMapVariant: rubricValueToSelect(source?.rubric?.defines_map_variant),
      sourceReviewDistinguishesProof: rubricValueToSelect(source?.rubric?.distinguishes_empirical_from_proof),
      sourceReviewProvesDescent: rubricValueToSelect(source?.rubric?.proves_descent),
      sourceReviewProvesCycleExclusion: rubricValueToSelect(source?.rubric?.proves_cycle_exclusion),
      sourceReviewUsesStatisticalArgument: rubricValueToSelect(source?.rubric?.uses_statistical_argument),
      sourceReviewValidationBacked: rubricValueToSelect(source?.rubric?.validation_backed)
    };
  }

  function buildReviewRubricPayload() {
    return {
      peer_reviewed: selectToRubricValue(quickForms.sourceReviewPeerReviewed),
      acknowledged_errors: selectToRubricValue(quickForms.sourceReviewAcknowledgedErrors),
      defines_map_variant: selectToRubricValue(quickForms.sourceReviewDefinesMapVariant),
      distinguishes_empirical_from_proof: selectToRubricValue(quickForms.sourceReviewDistinguishesProof),
      proves_descent: selectToRubricValue(quickForms.sourceReviewProvesDescent),
      proves_cycle_exclusion: selectToRubricValue(quickForms.sourceReviewProvesCycleExclusion),
      uses_statistical_argument: selectToRubricValue(quickForms.sourceReviewUsesStatisticalArgument),
      validation_backed: selectToRubricValue(quickForms.sourceReviewValidationBacked),
      notes: quickForms.sourceReviewNotes
    };
  }

  useEffect(() => {
    if (!selectedOrbitRun) {
      setOrbitFrame(0);
      return;
    }
    setOrbitFrame(0);
    const timer = window.setInterval(() => {
      setOrbitFrame((current) => {
        if (orbitFrames.length === 0) {
          return 0;
        }
        return (current + 1) % orbitFrames.length;
      });
    }, 550);
    return () => {
      window.clearInterval(timer);
    };
  }, [selectedOrbitRun?.id, orbitFrames.length]);

  useEffect(() => {
    if (state.runs.length === 0) {
      if (orbitRunId) {
        setOrbitRunId("");
      }
      return;
    }
    const selectedExists = state.runs.some((run) => run.id === orbitRunId);
    if (!selectedExists) {
      const fallback = state.runs.find((run) => run.status === "running") || state.runs[0];
      if (fallback && fallback.id !== orbitRunId) {
        setOrbitRunId(fallback.id);
      }
    }
  }, [state.runs, orbitRunId]);

  useEffect(() => {
    if (state.sources.length === 0) {
      if (quickForms.sourceReviewId) {
        setQuickForms((current) => ({ ...current, ...reviewFormPatch(null) }));
      }
      return;
    }
    const selectedExists = state.sources.some((source) => source.id === quickForms.sourceReviewId);
    if (!selectedExists) {
      const fallback = state.sources[0];
      setQuickForms((current) => ({
        ...current,
        ...reviewFormPatch(fallback)
      }));
    }
  }, [state.sources, quickForms.sourceReviewId]);

  useEffect(() => {
    if (state.claims.length === 0 && state.runs.length === 0) {
      return;
    }
    setQuickForms((current) => {
      const nextClaimId =
        current.linkClaimId && state.claims.some((claim) => claim.id === current.linkClaimId)
          ? current.linkClaimId
          : selectedEvidenceClaim?.id || state.claims[0]?.id || "";
      const nextRunId =
        current.linkRunId && state.runs.some((run) => run.id === current.linkRunId)
          ? current.linkRunId
          : selectedEvidenceRun?.id || validatedRuns[0]?.id || state.runs[0]?.id || "";

      if (nextClaimId === current.linkClaimId && nextRunId === current.linkRunId) {
        return current;
      }
      return {
        ...current,
        linkClaimId: nextClaimId,
        linkRunId: nextRunId
      };
    });
  }, [state.claims, state.runs, selectedEvidenceClaim?.id, selectedEvidenceRun?.id, validatedRuns]);

  useEffect(() => {
    const hasSelectedItem =
      (selectedEvidence.kind === "run" && state.runs.some((run) => run.id === selectedEvidence.id)) ||
      (selectedEvidence.kind === "claim" && state.claims.some((claim) => claim.id === selectedEvidence.id)) ||
      (selectedEvidence.kind === "artifact" && state.artifacts.some((artifact) => artifact.id === selectedEvidence.id));

    if (hasSelectedItem) {
      return;
    }

    const fallback =
      (validatedRuns[0] && { kind: "run", id: validatedRuns[0].id }) ||
      (filteredClaims[0] && { kind: "claim", id: filteredClaims[0].id }) ||
      (filteredRuns[0] && { kind: "run", id: filteredRuns[0].id }) ||
      (filteredArtifacts[0] && { kind: "artifact", id: filteredArtifacts[0].id }) ||
      null;

    if (fallback) {
      setSelectedEvidence(fallback);
    }
  }, [selectedEvidence.kind, selectedEvidence.id, state.runs, state.claims, state.artifacts, validatedRuns, filteredClaims, filteredRuns, filteredArtifacts]);

  function reveal(key, step) {
    setVisible((current) => ({ ...current, [key]: current[key] + step }));
  }

  function updateFilter(key, value) {
    setFilters((current) => ({ ...current, [key]: value }));
  }

  function clearRunFilters() {
    setFilters((current) => ({
      ...current,
      runQuery: "",
      runStatus: "all",
      runHardware: "all"
    }));
  }

  function clearTaskFilters() {
    setFilters((current) => ({
      ...current,
      taskQuery: "",
      taskStatus: "all"
    }));
  }

  function clearClaimFilters() {
    setFilters((current) => ({
      ...current,
      claimQuery: ""
    }));
  }

  function clearSourceFilters() {
    setFilters((current) => ({
      ...current,
      sourceQuery: "",
      sourceStatus: "all"
    }));
  }

  function clearArtifactFilters() {
    setFilters((current) => ({
      ...current,
      artifactQuery: ""
    }));
  }

  function updateQuickForm(key, value) {
    setQuickForms((current) => ({ ...current, [key]: value }));
  }

  function openEvidence(kind, id) {
    setSelectedEvidence({ kind, id });
    if (kind === "artifact") {
      setPreviewArtifactId(id);
    }
  }

  async function previewArtifact(artifactId) {
    setPreviewArtifactId(artifactId);
    if (artifactPreviews[artifactId]) {
      return;
    }
    try {
      const payload = await readJson(endpoints.artifactContent(artifactId));
      setArtifactPreviews((current) => ({ ...current, [artifactId]: payload }));
    } catch (error) {
      setActionState({
        pendingKey: "",
        tone: "bad",
        message: error.message || `Could not preview artifact ${artifactId}.`,
      });
    }
  }

  function selectSourceForReview(sourceId) {
    const source = state.sources.find((item) => item.id === sourceId);
    setQuickForms((current) => ({
      ...current,
      ...reviewFormPatch(source || null)
    }));
  }

  function addFallacyTag(field, tag) {
    setQuickForms((current) => ({
      ...current,
      [field]: appendCsvTag(current[field], tag)
    }));
  }

  function selectOrbitRun(runId) {
    setOrbitRunId(runId);
  }

  function jumpToSection(sectionId) {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ block: "start" });
    }
  }

  function focusEvidenceSection(sectionId, fallbackKind, fallbackItems) {
    jumpToSection(sectionId);
    const first = fallbackItems?.[0];
    if (first) {
      setSelectedEvidence({ kind: fallbackKind, id: first.id });
    }
  }

  function selectTab(tabId) {
    setActiveTab(tabId);
    setNavOpen(false);
  }

  async function runAction(actionKey, operation, onSuccess) {
    setActionState({
      pendingKey: actionKey,
      tone: "",
      message: ""
    });
    try {
      const result = await operation();
      if (onSuccess) {
        onSuccess(result);
      }
      setActionState({
        pendingKey: "",
        tone: "success",
        message: result?.id
          ? `${actionKey} saved as ${result.id}.`
          : `${actionKey} completed successfully.`
      });
      setRefreshNonce((current) => current + 1);
      return result;
    } catch (error) {
      setActionState({
        pendingKey: "",
        tone: "error",
        message: error.message
      });
      return null;
    }
  }

  async function handleRunSubmit(event) {
    event.preventDefault();
    await runAction(
      "queue run",
      () =>
        postJson(endpoints.runs, {
          direction_slug: quickForms.runDirection,
          name: quickForms.runName,
          range_start: Number(quickForms.runStart),
          range_end: Number(quickForms.runEnd),
          kernel: quickForms.runKernel,
          hardware: quickForms.runHardware,
          enqueue_only: true
        }),
      () => {
        setQuickForms((current) => ({
          ...current,
          runName: ""
        }));
      }
    );
  }

  async function handleTaskSubmit(event) {
    event.preventDefault();
    await runAction(
      "task",
      () =>
        postJson(endpoints.tasks, {
          direction_slug: quickForms.taskDirection,
          title: quickForms.taskTitle,
          kind: quickForms.taskKind,
          description: quickForms.taskDescription
        }),
      () => {
        setQuickForms((current) => ({
          ...current,
          taskTitle: "",
          taskDescription: ""
        }));
      }
    );
  }

  async function handleClaimSubmit(event) {
    event.preventDefault();
    await runAction(
      "claim",
      () =>
        postJson(endpoints.claims, {
          direction_slug: quickForms.claimDirection,
          title: quickForms.claimTitle,
          statement: quickForms.claimStatement
        }),
      () => {
        setQuickForms((current) => ({
          ...current,
          claimTitle: "",
          claimStatement: ""
        }));
      }
    );
  }

  async function handleClaimRunLinkSubmit(event) {
    event.preventDefault();
    if (!quickForms.linkClaimId || !quickForms.linkRunId) {
      setActionState({
        pendingKey: "",
        tone: "error",
        message: "Select both a claim and a run before creating a link."
      });
      return;
    }
    await runAction(
      "claim link",
      () =>
        postJson(endpoints.linkClaimRun, {
          claim_id: quickForms.linkClaimId,
          run_id: quickForms.linkRunId,
          relation: quickForms.linkRelation
        }),
      () => {
        setSelectedEvidence({ kind: "claim", id: quickForms.linkClaimId });
      }
    );
  }

  async function handleReviewSubmit(event) {
    event.preventDefault();
    await runAction(
      "direction review",
      () => postJson(`${apiBase}/api/directions/${quickForms.reviewDirection}/review`),
      (result) => {
        setReviewResult(result);
      }
    );
  }

  async function handleSourceSubmit(event) {
    event.preventDefault();
    await runAction(
      "source",
      () =>
        postJson(endpoints.sources, {
          direction_slug: quickForms.sourceDirection,
          title: quickForms.sourceTitle,
          authors: quickForms.sourceAuthors,
          year: quickForms.sourceYear,
          url: quickForms.sourceUrl,
          source_type: quickForms.sourceType,
          claim_type: quickForms.sourceClaimType,
          map_variant: quickForms.sourceMapVariant,
          summary: quickForms.sourceSummary,
          fallacy_tags: parseCsvList(quickForms.sourceTags)
        }),
      (result) => {
        setQuickForms((current) => ({
          ...current,
          sourceTitle: "",
          sourceAuthors: "",
          sourceUrl: "",
          sourceSummary: "",
          sourceTags: "",
          sourceMapVariant: "unspecified",
          ...reviewFormPatch(result || null)
        }));
      }
    );
  }

  async function handleRedditImport(post) {
    if (!post?.id) {
      return;
    }
    const claimType = post.signal === "review" ? "proof_attempt" : "discussion";
    await runAction(
      `reddit-${post.id}`,
      () =>
        postJson(endpoints.sources, {
          direction_slug: "lemma-workspace",
          title: post.title,
          authors: `u/${post.author}`,
          year: post.created_at ? String(new Date(post.created_at).getFullYear()) : "2026",
          url: post.permalink,
          source_type: "forum",
          claim_type: claimType,
          map_variant: "unspecified",
          summary: post.excerpt,
          notes: `Imported from r/Collatz feed on ${new Date().toISOString()}.`
        }),
      (result) => {
        setActiveTab("evidence");
        setQuickForms((current) => ({
          ...current,
          ...reviewFormPatch(result || null)
        }));
      }
    );
  }

  async function handleSourceReviewSubmit(event) {
    event.preventDefault();
    if (!quickForms.sourceReviewId) {
      setActionState({
        pendingKey: "",
        tone: "error",
        message: "Select a source before submitting a review."
      });
      return;
    }
    await runAction(
      "source review",
      () =>
        postJson(`${endpoints.sources}/${quickForms.sourceReviewId}/review`, {
          review_status: quickForms.sourceReviewStatus,
          map_variant: quickForms.sourceReviewMapVariant,
          notes: quickForms.sourceReviewNotes,
          fallacy_tags: parseCsvList(quickForms.sourceReviewTags),
          rubric: buildReviewRubricPayload()
        }),
      (result) => {
        setSourceReviewResult(result);
        setQuickForms((current) => ({
          ...current,
          ...reviewFormPatch(result || null)
        }));
      }
    );
  }

  async function handleProbeSubmit(event) {
    event.preventDefault();
    await runAction(
      "modular probe",
      () =>
        postJson(endpoints.modularProbe, {
          modulus: Number(quickForms.probeModulus),
          allowed_residues: parseCsvList(quickForms.probeResidues).map((value) => Number(value)),
          search_limit: Number(quickForms.probeLimit)
        }),
      (result) => {
        setProbeResult(result);
      }
    );
  }

  return (
    <main className={navOpen ? "page-shell app-shell nav-open" : "page-shell app-shell"}>
      <aside className="sidebar" aria-label="Dashboard navigation">
        <div className="sidebar-brand">
          <p className="eyebrow">Collatz Lab</p>
          <strong>Research shell</strong>
        </div>
        <div className="sidebar-section">
          <span className="sidebar-kicker">Live run</span>
          <label className="run-select">
            <span>Active orbit</span>
            <select
              value={orbitRunId || selectedOrbitRun?.id || ""}
              onChange={(event) => setOrbitRunId(event.target.value)}
              aria-label="Select active run"
            >
              {selectorRunOptions.length === 0 ? <option value="">No runs available</option> : null}
              {selectorRunOptions.map((run) => (
                <option key={run.id} value={run.id}>
                  {run.id} {run.status === "running" ? "- live now" : run.status ? `- ${run.status}` : ""}
                </option>
              ))}
            </select>
            <span className="run-select-hint">
              {runningRunOptions.length > 0
                ? selectedRunIsLive
                  ? "The selector is currently focused on real active runs."
                  : "Active runs are prioritized here; the pinned replay is kept visible until you switch."
                : "No live runs right now. The selector falls back to saved history."}
            </span>
          </label>
        </div>
        <div className="sidebar-section sidebar-runtime">
          <span className="sidebar-kicker">Runtime now</span>
          <div className="sidebar-runtime-grid">
            <article className="sidebar-runtime-card">
              <span>CPU now</span>
              <strong>{cpuNowLabel}</strong>
            </article>
            <article className="sidebar-runtime-card">
              <span>GPU now</span>
              <strong>{gpuNowLabel}</strong>
            </article>
            <article className="sidebar-runtime-card">
              <span>Active runs</span>
              <strong>{runningRuns}</strong>
            </article>
            <article className="sidebar-runtime-card">
              <span>Queued</span>
              <strong>{queuedRuns}</strong>
            </article>
          </div>
          <p className="sidebar-note">Live machine usage and queue pressure from the local lab.</p>
        </div>
        <nav className="sidebar-nav" aria-label="Dashboard sections">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              type="button"
              className={tab.id === activeTab ? "sidebar-link active" : "sidebar-link"}
              onClick={() => selectTab(tab.id)}
            >
              <span>{tab.label}</span>
              <small>{tab.id === activeTab ? "Open" : "View"}</small>
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">
          <div>
            <span>Mode</span>
            <strong>{activeTabLabel}</strong>
          </div>
          <button type="button" className="secondary-button sidebar-refresh" onClick={() => setRefreshNonce((current) => current + 1)}>
            Refresh
          </button>
        </div>
      </aside>
      {navOpen ? <button type="button" className="nav-backdrop" aria-label="Close navigation" onClick={() => setNavOpen(false)} /> : null}
      <div className="main-column">
        <header className="mobile-topbar">
          <button
            type="button"
            className="menu-button"
            aria-label={navOpen ? "Close navigation" : "Open navigation"}
            onClick={() => setNavOpen((current) => !current)}
          >
            <span />
            <span />
            <span />
          </button>
          <div className="mobile-topbar-copy">
            <span>Collatz Lab</span>
            <strong>{activeTabLabel}</strong>
          </div>
          <button type="button" className="secondary-button mobile-refresh" onClick={() => setRefreshNonce((current) => current + 1)}>
            Refresh
          </button>
        </header>

        {state.error ? <section className="error-banner">{state.error}</section> : null}
        {state.loading ? <section className="loading-banner">Loading real data from the local API...</section> : null}

        <div className="workspace-shell">
        <div className="workspace-primary">

        {activeTab === "overview" ? (
        <section className="tab-panel">
          <section className="hero">
            <div className="hero-copy-block">
              <p className="eyebrow">Collatz Lab</p>
              <h1>T(n) = n/2 or 3n+1</h1>
              <p className="hero-copy">
                Real runs, claims, and live math. Queue work, inspect evidence, and follow every Collatz step
                from actual checkpoints.
              </p>
              <div className="formula-banner">
                <span>Collatz operator</span>
                <strong>T(n) = n / 2 for even n, and T(n) = 3n + 1 for odd n.</strong>
              </div>
            </div>
            <div className="hero-meta">
              <div className="hero-spotlight">
                <span className="spotlight-kicker">Current state</span>
                <strong>{hasLoaded ? `${summary.run_count} runs tracked` : "Loading..."}</strong>
                <p>
                  {hasLoaded
                    ? `${summary.validated_run_count} validated, ${summary.worker_count ?? state.workers.length} workers, ${supportedClaims} supported claims.`
                    : "Connecting to API..."}
                </p>
              </div>
              <p className="diagnostic-line">
                API {apiBase} | Poll {pollIntervalMs / 1000}s | {state.lastUpdated || "pending"}
              </p>
            </div>
          </section>

          {hasLoaded ? (
            <section className="summary-strip">
              <SummaryCard label="Directions" value={summary.direction_count} note="parallel tracks" />
              <SummaryCard label="Runs" value={summary.run_count} note="compute records" />
              <SummaryCard label="Validated" value={summary.validated_run_count} note="replayed runs" />
              <SummaryCard label="Workers" value={summary.worker_count ?? state.workers.length} note={`${summary.active_worker_count ?? liveWorkers.length} active`} />
              <SummaryCard label="Claims" value={supportedClaims} note="with evidence" />
              <SummaryCard label="Tasks" value={summary.open_task_count} note="open items" />
              <SummaryCard label="Artifacts" value={summary.artifact_count} note="outputs" />
            </section>
          ) : null}

          <SectionIntro
            title="Start here"
            text="Understand the lab at a glance: what is happening, what the words mean, and where to go next."
          />
          <div className="start-strip">
            <article className="start-card">
              <span>1</span>
              <strong>Queue a run</strong>
              <p>Create a real interval experiment in Operations.</p>
            </article>
            <article className="start-card">
              <span>2</span>
              <strong>Let a worker claim it</strong>
              <p>The worker moves a queued run into active execution.</p>
            </article>
            <article className="start-card">
              <span>3</span>
              <strong>Inspect evidence</strong>
              <p>Read runs, claims, and artifacts before trusting a pattern.</p>
            </article>
            <article className="start-card">
              <span>4</span>
              <strong>Review the track</strong>
              <p>Promote, freeze, or refute only after evidence lands.</p>
            </article>
          </div>
          <Legend />
          {!hasEvidence ? (
            <section className="loading-banner">
              Only seeded lab metadata exists so far. Real experimental evidence will appear here after runs, claims, tasks, or artifacts are created.
            </section>
          ) : null}
          {hasLoaded ? (
            <div className="overview-grid">
              <article className="panel">
                <SectionIntro
                  title="Current state"
                  text="This is the operational snapshot: what is queued, what is active, and whether validation is keeping up."
                  action={
                    <button className="secondary-button" type="button" onClick={() => setActiveTab("live-math")}>
                      Open Live Math
                    </button>
                  }
                />
                <div className="capability-grid">
                  <CapabilityCard label="API" value="live" note={`Connected to ${apiBase}`} />
                  <CapabilityCard label="Queued runs" value={queuedRuns} note="waiting for a worker slot" />
                  <CapabilityCard label="Live runs" value={runningRuns} note="currently executing experiment records" />
                  <CapabilityCard label="CPU runs" value={cpuRuns} note={cpuRuns > 0 ? "observed in the ledger" : "not recorded yet"} />
                  <CapabilityCard label="GPU runs" value={gpuRuns} note={gpuRuns > 0 ? "GPU path exercised" : "no GPU run has been recorded yet"} />
                  <CapabilityCard label="Validated" value={validatedRunCount} note="direct and accelerated replay have matched" />
                  <CapabilityCard label="Mixed" value={mixedRuns} note={mixedRuns > 0 ? "hybrid hardware entries exist" : "no mixed runs yet"} />
                </div>
                <div className="surface-split">
                  <article className="surface-card">
                    <span className="surface-kicker">Hardware inventory</span>
                    {hardwareInventory.length === 0 ? (
                      <p>No hardware inventory endpoint is visible yet. The run ledger still gives us CPU/GPU evidence.</p>
                    ) : (
                      <div className="surface-list">
                        {hardwareInventory.slice(0, 4).map((item, index) => {
                          const label = item.name || item.label || item.model || item.type || item.id || `Hardware ${index + 1}`;
                          const details = [
                            item.status,
                            item.kind,
                            item.architecture,
                            item.metadata?.memory_mib ? `${item.metadata.memory_mib} MiB` : null,
                            item.memory_gb ? `${item.memory_gb} GB` : null,
                            item.memory ? `${item.memory} GB` : null,
                            Array.isArray(item.supported_kernels) && item.supported_kernels.length > 0
                              ? item.supported_kernels.join(", ")
                              : "detected only"
                          ]
                            .filter(Boolean)
                            .join(" | ");
                          return (
                            <div key={item.id || `${label}-${index}`} className="surface-row">
                              <strong>{label}</strong>
                              <p>{details || "Inventory record available"}</p>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </article>
                  <article className="surface-card">
                    <span className="surface-kicker">Live workers</span>
                    {liveWorkers.length === 0 ? (
                      <p>No live worker registry is visible yet. Start the managed worker and the queue will become executable.</p>
                    ) : (
                      <div className="surface-list">
                        {liveWorkers.slice(0, 4).map((worker, index) => {
                          const label = worker.name || worker.label || worker.id || worker.host || `Worker ${index + 1}`;
                          const details = [
                            worker.status || worker.state,
                            worker.hardware,
                            worker.current_run_id ? `run ${worker.current_run_id}` : "idle",
                            worker.last_heartbeat_at || worker.last_seen_at
                          ]
                            .filter(Boolean)
                            .join(" | ");
                          return (
                            <div key={worker.id || `${label}-${index}`} className="surface-row">
                              <strong>{label}</strong>
                              <p>{details || "Worker is registered"}</p>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </article>
                </div>
                <div className="note-block overview-callout">
                  <p>
                    The live formulas, checkpoint tape, and record events now live in their own page so they no longer compete with the overview layout.
                  </p>
                  <button className="secondary-button" type="button" onClick={() => setActiveTab("live-math")}>
                    Go To Live Math
                  </button>
                </div>
              </article>

              <article className="panel">
                <SectionIntro
                  title="Tracks at a glance"
                  text="These are the current research tracks and whether they are active, promising, or stalled."
                />
                <div className="tracks-grid">
                  {visibleDirections.map((direction) => {
                    const guide = directionGuide[direction.slug];
                    return (
                    <article key={direction.slug} className="list-card">
                      <div className="card-head">
                        <div>
                          <h3>{direction.title}</h3>
                          <p>{direction.slug}</p>
                        </div>
                        <StatusPill value={direction.status} />
                      </div>
                      {guide ? <span className={`evidence-type-pill evidence-type-${direction.slug}`}>{guide.label}</span> : null}
                      <p>{direction.description}</p>
                      {guide ? <p className="meta-line"><strong>{guide.caution}</strong> {guide.role}</p> : null}
                    </article>
                  )})}
                </div>
              </article>

              <article className="panel">
                <SectionIntro
                  title="Consensus baseline"
                  text="External sources are reviewed against this baseline so the lab does not confuse partial progress with a proof."
                  action={<span className="filter-count">{reviewedSources} reviewed sources</span>}
                />
                {state.baseline ? (
                  <>
                    <div className="capability-grid">
                      <CapabilityCard label="Problem status" value={prettyLabel(state.baseline.problem_status)} note={state.baseline.checked_as_of} />
                      <CapabilityCard label="Verified up to" value={state.baseline.verified_up_to} note="computational evidence only" />
                      <CapabilityCard label="Flagged sources" value={flaggedSources} note="need stronger review or counterexamples" />
                    </div>
                    <div className="baseline-note-block">
                      <span className="surface-kicker">Review rule</span>
                      <p>{state.baseline.note}</p>
                    </div>
                    <div className="baseline-grid">
                      {state.baseline.items.map((item) => (
                        <article key={item.title} className="list-card baseline-item-card">
                          <div className="card-head">
                            <div>
                              <span className="surface-kicker">Baseline item</span>
                              <strong>{item.title}</strong>
                            </div>
                            <span className="filter-count">{baselineBadge(item.title)}</span>
                          </div>
                          <p>{item.detail}</p>
                          <a
                            className="baseline-source-link"
                            href={item.source_url}
                            target="_blank"
                            rel="noreferrer"
                          >
                            <span>{compactSourceLabel(item.source_url)}</span>
                            <strong>Open source</strong>
                          </a>
                        </article>
                      ))}
                    </div>
                  </>
                ) : (
                  <EmptyState title="No baseline yet" text="The API has not exposed the consensus baseline payload." />
                )}
              </article>
            </div>
          ) : (
            <section className="loading-banner">Waiting for real data from the local API...</section>
          )}
        </section>
        ) : null}

        {activeTab === "live-math" ? (
        <section className="tab-panel live-math-page">
          <MathTicker run={selectedOrbitRun} orbit={orbitFrames} frameIndex={orbitFrame} />
          <SectionIntro
            title="Live Math"
            text="This page is the calculation room: formulas, worker checkpoint, and record tape are all visible in one place."
            action={<span className="filter-count">{selectedOrbitRun ? selectedOrbitRun.id : "no run selected"}</span>}
          />
          <div className={selectedRunIsLive ? "runtime-banner runtime-banner-live" : "runtime-banner runtime-banner-idle"}>
            {selectedOrbitRun ? (
              selectedRunIsLive ? (
                `Pinned run ${selectedOrbitRun.id} is live now. The formulas below follow worker checkpoints from an active computation.`
              ) : anyLiveRun ? (
                `There are live runs in the lab, but ${selectedOrbitRun.id} is ${selectedOrbitRun.status}. The formulas below replay saved checkpoints from a finished run.`
              ) : (
                `No Collatz run is active right now. ${selectedOrbitRun.id} is ${selectedOrbitRun.status}, so the formulas below are historical replay, not live compute.`
              )
            ) : (
              "No run is pinned yet. Queue a run from Operations to produce live checkpoints."
            )}
          </div>
          <LiveMathNavigator
            runs={selectorRunOptions}
            selectedRun={selectedOrbitRun}
            onSelectRun={selectOrbitRun}
            onJumpToSection={jumpToSection}
          />
          <div className="live-math-strip">
            <article className="live-status-card">
              <span className="metric-label">Selected run</span>
              <strong>{selectedOrbitRun?.id ?? "none"}</strong>
              <p>{selectedOrbitRun?.summary || "Pick a run to inspect the mathematical trace."}</p>
            </article>
            <article className="live-status-card">
              <span className="metric-label">Run progress</span>
              <strong>
                {selectedRunProgress.total > 0
                  ? `${selectedRunProgress.processed.toLocaleString()} / ${selectedRunProgress.total.toLocaleString()}`
                  : "No progress yet"}
              </strong>
              <p>
                {selectedRunIsLive
                  ? `${selectedRunProgress.percent.toFixed(1)}% of the interval has been processed live.`
                  : selectedOrbitRun
                    ? `Saved result: ${selectedRunProgress.percent.toFixed(1)}% of the interval was processed before completion.`
                    : "No run progress is available yet."}
              </p>
            </article>
            <article className="live-status-card">
              <span className="metric-label"><span className={activeCpuRuns > 0 ? "live-dot" : "idle-dot"} /> CPU execution</span>
              <strong>
                {cpuCapability?.metadata?.logical_cores
                  ? `${cpuCapability.metadata.logical_cores} logical cores`
                  : "CPU visible"}
              </strong>
              {cpuCapability?.metadata?.usage_percent != null ? (
                <>
                  <div className="usage-bar-wrap">
                    <div className="usage-bar">
                      <div className="usage-bar-fill" style={{ width: `${Math.min(100, cpuCapability.metadata.usage_percent)}%` }} />
                    </div>
                    <span className="usage-pct">{cpuCapability.metadata.usage_percent.toFixed(1)}%</span>
                  </div>
                  <p className="live-card-note">
                    Host CPU usage now. Collatz CPU runs active now: {activeCpuRuns}. Historical CPU runs saved: {cpuRuns}.
                  </p>
                </>
              ) : (
                <>
                  {cpuPressure ? (
                    <div className="pressure-row">
                      <span className={`pressure-badge pressure-${cpuPressure.state}`}>{cpuPressure.state}</span>
                      <span className="pressure-label">browser pressure only</span>
                    </div>
                  ) : null}
                  <div className="live-hw-detail">
                    <span className={activeCpuRuns > 0 ? "live-dot-sm" : "idle-dot-sm"} />
                    <span>Collatz CPU runs active now: {activeCpuRuns}. Historical CPU runs saved: {cpuRuns}.</span>
                  </div>
                </>
              )}
            </article>
            <article className="live-status-card">
              <span className="metric-label"><span className={activeGpuRuns > 0 ? "live-dot" : "idle-dot"} /> GPU execution</span>
              <strong>{gpuCapability?.label || "No GPU record"}</strong>
              {gpuCapability?.metadata?.usage_percent != null ? (
                <>
                  <div className="usage-bar-wrap">
                    <div className="usage-bar">
                      <div className="usage-bar-fill usage-bar-fill-gpu" style={{ width: `${Math.min(100, gpuCapability.metadata.usage_percent)}%` }} />
                    </div>
                    <span className="usage-pct">{gpuCapability.metadata.usage_percent.toFixed(1)}%</span>
                  </div>
                  <p className="live-card-note">
                    Host GPU usage now. Collatz GPU runs active now: {activeGpuRuns}. Historical GPU runs saved: {gpuRuns}.
                    {gpuCapability?.metadata?.memory_used_mib != null && gpuCapability?.metadata?.memory_total_mib != null
                      ? ` VRAM ${Math.round(gpuCapability.metadata.memory_used_mib)} / ${Math.round(gpuCapability.metadata.memory_total_mib)} MiB.`
                      : ""}
                  </p>
                </>
              ) : (
                <div className="live-hw-detail">
                  <span className={activeGpuRuns > 0 ? "live-dot-sm" : "idle-dot-sm"} />
                  <span>
                    {gpuCapability
                      ? `Collatz GPU runs active now: ${activeGpuRuns}. Historical GPU runs saved: ${gpuRuns}.`
                      : "No GPU capability is exposed."}
                  </span>
                </div>
              )}
            </article>
          </div>
          <div className="live-math-shell">
            <RunRail runs={runRailOptions} selectedRunId={selectedOrbitRun?.id ?? ""} onSelectRun={selectOrbitRun} />
            <OrbitPanel
              sectionId="live-trace"
              run={selectedOrbitRun}
              worker={selectedOrbitWorker}
              frameIndex={orbitFrame}
              onSelectRun={selectOrbitRun}
              runs={selectorRunOptions}
              expanded
              showSelector={false}
            />
          </div>
          <div className="live-support-grid">
            <article className="panel" id="live-ledger">
              <SectionIntro
                title="Pinned run ledger"
                text="This is the current run under inspection and the exact interval that the math page is following."
              />
              {selectedOrbitRun ? (
                <div className="stack-list">
                  <article className="list-card live-mini-run">
                    <div className="card-head">
                      <div>
                        <h3>{selectedOrbitRun.id}</h3>
                        <p>{selectedOrbitRun.name}</p>
                      </div>
                      <StatusPill value={selectedOrbitRun.status} />
                    </div>
                    <p>
                      {selectedOrbitRun.range_start} - {selectedOrbitRun.range_end} | {selectedOrbitRun.kernel} | {selectedOrbitRun.hardware}
                    </p>
                    <p className="meta-line">
                      checkpoint {selectedOrbitRun.checkpoint?.last_processed ?? "unknown"} | next {selectedOrbitRun.checkpoint?.next_value ?? "unknown"}
                    </p>
                  </article>
                </div>
              ) : (
                <EmptyState title="No run pinned" text="Pick a run in the navigator rail to inspect it here." />
              )}
            </article>
            <article className="panel" id="live-records">
              <SectionIntro
                title="Execution surface"
                text="These values are taken from the current API snapshot, so the panel stays honest even when the registry is small."
              />
              <div className="stack-list compact-stack">
                <article className="list-card">
                  <strong>{liveWorkers.length} live workers</strong>
                  <p>
                    {liveWorkers.length > 0
                      ? `Visible workers: ${liveWorkers
                          .slice(0, 3)
                          .map((worker) => worker.name || worker.label || worker.id || "worker")
                          .join(", ")}`
                      : "No live worker registry is visible yet."}
                  </p>
                </article>
                <article className="list-card">
                  <strong>{hardwareInventory.length} hardware records</strong>
                  <p>
                    {gpuCapability?.label || cpuCapability?.label
                      ? `Capability ledger includes ${[cpuCapability?.label, gpuCapability?.label].filter(Boolean).join(" and ")}.`
                      : "No hardware capability record is visible yet."}
                  </p>
                </article>
                <article className="list-card">
                  <strong>Pinned worker state</strong>
                  <p>
                    {selectedOrbitWorker
                      ? `${selectedOrbitWorker.name || selectedOrbitWorker.id || "worker"} is ${selectedOrbitWorker.status || "unknown"} on ${selectedOrbitWorker.hardware || "unknown hardware"}.`
                      : "No worker is currently matched to the pinned run."}
                  </p>
                </article>
                <article className="list-card">
                  <strong>Selected kernel path</strong>
                  <p>
                    {selectedOrbitRun
                      ? `${selectedOrbitRun.kernel} on ${selectedOrbitRun.hardware} is the real run path for this view.`
                      : "No real run path is pinned yet."}
                  </p>
                </article>
              </div>
            </article>
          </div>
        </section>
        ) : null}

        {activeTab === "directions" ? (
        <section className="tab-panel">
          <SectionIntro
            title="Research directions"
            text="Each direction should remain understandable on its own: what it is testing, who owns it, and what counts as success or abandonment."
            action={
              <ShowMoreButton
                total={state.directions.length}
                visible={visible.directions}
                label="directions"
                onClick={() => reveal("directions", 3)}
              />
            }
          />
          {!hasLoaded ? (
            <section className="loading-banner">Waiting for real direction data...</section>
          ) : state.directions.length === 0 ? (
            <EmptyState title="No directions yet" text="Seed the lab or create a direction in the backend first." />
          ) : (
            <div className="stack-list">
              {visibleDirections.map((direction) => (
                <article key={direction.slug} className="panel direction-panel">
                  <div className="card-head">
                    <div>
                      <h3>{direction.title}</h3>
                      <p>{direction.slug}</p>
                    </div>
                    <StatusPill value={direction.status} />
                  </div>
                  <p>{direction.description}</p>
                  {directionGuide[direction.slug] ? (
                    <div className="direction-guide-grid">
                      <div>
                        <span className={`evidence-type-pill evidence-type-${direction.slug}`}>{directionGuide[direction.slug].label}</span>
                      </div>
                      <div className="note-block">
                        <p><strong>Role</strong> {directionGuide[direction.slug].role}</p>
                        <p><strong>Caution</strong> {directionGuide[direction.slug].caution}</p>
                      </div>
                    </div>
                  ) : null}
                  <div className="metric-grid">
                    <div>
                      <span className="metric-label">Owner</span>
                      <strong>{direction.owner}</strong>
                    </div>
                    <div>
                      <span className="metric-label">Score</span>
                      <strong>{direction.score}</strong>
                    </div>
                  </div>
                  <div className="note-block">
                    <p><strong>Success</strong> {direction.success_criteria}</p>
                    <p><strong>Abandon</strong> {direction.abandon_criteria}</p>
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>
        ) : null}

        {activeTab === "evidence" ? (
        <section className="tab-panel">
          <SectionIntro
            title="Evidence"
            text="Runs, claims, and artifacts are shown separately so you can inspect compute output without mixing it with proof notes."
          />
          <div className="note-block">
            <p><strong>Research stance</strong> Verification runs are evidence and falsification tools, not the proof strategy itself. The proof-facing work stays in claims, parity filters, inverse-tree structure, source review, and lemma testing.</p>
          </div>
          <div className="evidence-guide-grid">
            {evidenceGuide.map((item) => (
              <article key={item.kind} className="list-card evidence-guide-card">
                <span className={`evidence-type-pill evidence-type-${item.kind}`}>{item.title}</span>
                <p>{item.detail}</p>
              </article>
            ))}
          </div>
          <div className="evidence-summary-strip">
            <button
              className="summary-card evidence-summary-card summary-card-button"
              type="button"
              onClick={() => focusEvidenceSection("evidence-validated-results", "run", validatedRuns)}
            >
              <span>Validated results</span>
              <strong>{validatedRuns.length}</strong>
              <p>Independently replayed runs you can trust more than raw completions.</p>
            </button>
            <button
              className="summary-card evidence-summary-card summary-card-button"
              type="button"
              onClick={() => focusEvidenceSection("evidence-claims", "claim", filteredClaims)}
            >
              <span>Claims</span>
              <strong>{filteredClaims.length}</strong>
              <p>Mathematical statements, separate from compute logs and separate from proof attempts.</p>
            </button>
            <button
              className="summary-card evidence-summary-card summary-card-button"
              type="button"
              onClick={() => focusEvidenceSection("evidence-artifacts", "artifact", filteredArtifacts)}
            >
              <span>Run artifacts</span>
              <strong>{filteredArtifacts.length}</strong>
              <p>JSON outputs, validation reports, and note files available for preview or export.</p>
            </button>
          </div>
          <article className="panel">
            <SectionIntro
              title="Evidence tracking ledger"
              text="This ledger makes the distinction explicit: validated results, claims, artifacts, and raw runs are separate record types."
              action={
                <ShowMoreButton
                  total={evidenceLedger.length}
                  visible={visible.ledger}
                  label="ledger entries"
                  onClick={() => reveal("ledger", 6)}
                />
              }
            />
            {visibleEvidenceLedger.length === 0 ? (
              <EmptyState title="No evidence yet" text="Runs, claims, and artifacts will appear here as soon as they are saved." />
            ) : (
              <div className="stack-list">
                {visibleEvidenceLedger.map((item) => (
                  <article
                    key={item.key}
                    className={
                      selectedEvidence.kind === item.openKind && selectedEvidence.id === item.id
                        ? "list-card evidence-card-selected evidence-ledger-card"
                        : "list-card evidence-ledger-card"
                    }
                  >
                    <div className="card-head">
                      <div>
                        <div className="evidence-ledger-headline">
                          <span className={`evidence-type-pill evidence-type-${item.kind}`}>{prettyLabel(item.kind)}</span>
                          <StatusPill value={item.status} />
                        </div>
                        <h3>{item.title}</h3>
                        <p>{item.id} | {item.direction}</p>
                      </div>
                      <span className="filter-count">{formatTimestamp(item.timestamp)}</span>
                    </div>
                    <p>{item.summary}</p>
                    <p className="meta-line">{item.meta}</p>
                    <div className="card-action-row">
                      <button className="secondary-button" type="button" onClick={() => openEvidence(item.openKind, item.id)}>
                        Open in inspector
                      </button>
                      <button className="secondary-button" type="button" onClick={() => exportJsonFile(`${item.id}.json`, item)}>
                        Export summary
                      </button>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </article>
          <EvidenceDetailPanel
            selectedKind={selectedEvidence.kind}
            selectedRun={selectedEvidenceRun}
            selectedClaim={selectedEvidenceClaim}
            selectedArtifact={selectedEvidenceArtifact}
            relatedLinks={selectedEvidenceLinks}
            relatedRuns={relatedRuns}
            relatedClaims={relatedClaims}
            relatedArtifacts={relatedArtifacts}
            previewPayload={previewPayload}
            onSelectEvidence={openEvidence}
            onPreviewArtifact={previewArtifact}
            onExportJson={exportJsonFile}
            onExportText={exportTextFile}
          />
          <div className="tab-subgrid">
            <article className="panel" id="evidence-validated-results">
              <SectionIntro
                title="Validated results"
                text="These are the runs that passed an independent validation path. Start here before trusting any broader pattern."
              />
              {!hasLoaded ? (
                <section className="loading-banner">Waiting for validated run data...</section>
              ) : validatedRuns.length === 0 ? (
                <EmptyState title="No validated runs yet" text="Run a validation pass to create high-trust results." />
              ) : (
                <div className="stack-list">
                  {validatedRuns.map((run) => (
                    <article key={`validated-${run.id}`} className={selectedEvidence.kind === "run" && selectedEvidence.id === run.id ? "list-card evidence-card-selected" : "list-card"}>
                      <div className="card-head">
                        <div>
                          <span className="evidence-type-pill evidence-type-validated-result">Validated result</span>
                          <h3>{run.name}</h3>
                          <p>{run.id} | validated result</p>
                        </div>
                        <StatusPill value={run.status} />
                      </div>
                      <div className="metric-grid three-up">
                        <div>
                          <span className="metric-label">Direction</span>
                          <strong>{run.direction_slug}</strong>
                        </div>
                        <div>
                          <span className="metric-label">Interval</span>
                          <strong>{run.range_start} to {run.range_end}</strong>
                        </div>
                        <div>
                          <span className="metric-label">Validation</span>
                          <strong>Independent replay passed</strong>
                        </div>
                      </div>
                      <p className="meta-line">{run.summary || "No validation summary yet."}</p>
                      <div className="card-action-row">
                        <button className="secondary-button" type="button" onClick={() => openEvidence("run", run.id)}>
                          Open details
                        </button>
                        <button className="secondary-button" type="button" onClick={() => exportJsonFile(`${run.id}.json`, run)}>
                          Export JSON
                        </button>
                      </div>
                    </article>
                  ))}
                </div>
              )}
            </article>

            <article className="panel" id="evidence-runs">
              <SectionIntro
                title="Runs"
                text="Reproducible compute runs with interval, status, and top metrics."
                action={
                  <ShowMoreButton
                    total={filteredRuns.length}
                    visible={visible.runs}
                    label="runs"
                    onClick={() => reveal("runs", 6)}
                  />
                }
              />
              <FilterBar onClear={clearRunFilters}>
                <FilterField label="Search">
                  <input
                    className="filter-input"
                    value={filters.runQuery}
                    onChange={(event) => updateFilter("runQuery", event.target.value)}
                    placeholder="run, direction, kernel..."
                  />
                </FilterField>
                <FilterField label="Status">
                  <select
                    className="filter-input"
                    value={filters.runStatus}
                    onChange={(event) => updateFilter("runStatus", event.target.value)}
                  >
                    <option value="all">All</option>
                    <option value="queued">Queued</option>
                    <option value="running">Running</option>
                    <option value="completed">Completed</option>
                    <option value="validated">Validated</option>
                    <option value="failed">Failed</option>
                  </select>
                </FilterField>
                <FilterField label="Hardware">
                  <select
                    className="filter-input"
                    value={filters.runHardware}
                    onChange={(event) => updateFilter("runHardware", event.target.value)}
                  >
                    <option value="all">All</option>
                    <option value="cpu">CPU</option>
                    <option value="gpu">GPU</option>
                    <option value="mixed">Mixed</option>
                  </select>
                </FilterField>
              </FilterBar>
              {!hasLoaded ? (
                <section className="loading-banner">Waiting for real run data...</section>
              ) : filteredRuns.length === 0 ? (
                <EmptyState title="No runs match the filters" text="Clear the filters or start a new run from the CLI or API." />
              ) : (
                <div className="stack-list">
                  {visibleRuns.map((run) => (
                    <article key={run.id} className={selectedEvidence.kind === "run" && selectedEvidence.id === run.id ? "list-card evidence-card-selected" : "list-card"}>
                      <div className="card-head">
                        <div>
                          <span className={`evidence-type-pill ${run.status === "validated" ? "evidence-type-validated-result" : "evidence-type-run"}`}>
                            {run.status === "validated" ? "Validated result" : "Run"}
                          </span>
                          <strong>{run.id}</strong>
                        </div>
                        <StatusPill value={run.status} />
                      </div>
                      <h3>{run.name}</h3>
                      <div className="metric-grid three-up">
                        <div>
                          <span className="metric-label">Direction</span>
                          <strong>{run.direction_slug}</strong>
                        </div>
                        <div>
                          <span className="metric-label">Interval</span>
                          <strong>{run.range_start} to {run.range_end}</strong>
                        </div>
                        <div>
                          <span className="metric-label">Excursion</span>
                          <strong>{run.metrics?.max_excursion?.value ?? "-"}</strong>
                        </div>
                      </div>
                      <p className="meta-line">
                        {run.kernel} | {run.hardware} | {run.summary || "No summary yet"}
                      </p>
                      <div className="card-action-row">
                        <button className="secondary-button" type="button" onClick={() => openEvidence("run", run.id)}>
                          Open details
                        </button>
                        <button className="secondary-button" type="button" onClick={() => exportJsonFile(`${run.id}.json`, run)}>
                          Export JSON
                        </button>
                      </div>
                    </article>
                  ))}
                </div>
              )}
            </article>

            <article className="panel" id="evidence-claims">
              <SectionIntro
                title="Claims"
                text="Candidate mathematical statements and their current evidence status."
                action={
                  <ShowMoreButton
                    total={filteredClaims.length}
                    visible={visible.claims}
                    label="claims"
                    onClick={() => reveal("claims", 4)}
                  />
                }
              />
              <FilterBar onClear={clearClaimFilters} clearLabel="Clear search">
                <FilterField label="Search">
                  <input
                    className="filter-input"
                    value={filters.claimQuery}
                    onChange={(event) => updateFilter("claimQuery", event.target.value)}
                    placeholder="claim, direction, status..."
                  />
                </FilterField>
              </FilterBar>
              {!hasLoaded ? (
                <section className="loading-banner">Waiting for real claim data...</section>
              ) : filteredClaims.length === 0 ? (
                <EmptyState title="No claims match the search" text="Adjust the search or create a new claim." />
              ) : (
                <div className="stack-list">
                  {visibleClaims.map((claim) => (
                    <article key={claim.id} className={selectedEvidence.kind === "claim" && selectedEvidence.id === claim.id ? "list-card evidence-card-selected" : "list-card"}>
                      <div className="card-head">
                        <div>
                          <span className="evidence-type-pill evidence-type-claim">Claim</span>
                          <h3>{claim.title}</h3>
                          <p>{claim.id} | {claim.direction_slug}</p>
                        </div>
                        <StatusPill value={claim.status} />
                      </div>
                      <p>{claim.statement}</p>
                      <p className="meta-line">
                        Dependencies: {claim.dependencies?.length ? claim.dependencies.join(", ") : "none"}
                      </p>
                      <div className="card-action-row">
                        <button className="secondary-button" type="button" onClick={() => openEvidence("claim", claim.id)}>
                          Open details
                        </button>
                        <button className="secondary-button" type="button" onClick={() => exportJsonFile(`${claim.id}.json`, claim)}>
                          Export JSON
                        </button>
                      </div>
                    </article>
                  ))}
                </div>
              )}
            </article>

            <article className="panel" id="evidence-artifacts">
              <SectionIntro
                title="Artifacts"
                text="Files produced by the lab: notes, validation reports, and run outputs."
                action={
                  <ShowMoreButton
                    total={filteredArtifacts.length}
                    visible={visible.artifacts}
                    label="artifacts"
                    onClick={() => reveal("artifacts", 4)}
                  />
                }
              />
              <FilterBar onClear={clearArtifactFilters} clearLabel="Clear search">
                <FilterField label="Search">
                  <input
                    className="filter-input"
                    value={filters.artifactQuery}
                    onChange={(event) => updateFilter("artifactQuery", event.target.value)}
                    placeholder="path, checksum, kind..."
                  />
                </FilterField>
              </FilterBar>
              {!hasLoaded ? (
                <section className="loading-banner">Waiting for real artifact data...</section>
              ) : filteredArtifacts.length === 0 ? (
                <EmptyState title="No artifacts match the search" text="Reports and run outputs will appear here automatically." />
              ) : (
                <div className="stack-list">
                  {visibleArtifacts.map((artifact) => (
                    <article key={artifact.id} className={selectedEvidence.kind === "artifact" && selectedEvidence.id === artifact.id ? "list-card evidence-card-selected" : "list-card"}>
                      <div className="card-head">
                        <div>
                          <span className="evidence-type-pill evidence-type-artifact">Artifact</span>
                          <strong>{artifact.id}</strong>
                        </div>
                        <StatusPill value={artifact.kind} />
                      </div>
                      <p>{artifact.path}</p>
                      <p className="checksum">sha256: {artifact.checksum.slice(0, 18)}...</p>
                      <div className="card-action-row">
                        <button className="secondary-button" type="button" onClick={() => openEvidence("artifact", artifact.id)}>
                          Open details
                        </button>
                        <button className="secondary-button" type="button" onClick={() => previewArtifact(artifact.id)}>
                          Preview file
                        </button>
                        <a className="secondary-button detail-download-link" href={endpoints.artifactDownload(artifact.id)}>
                          Download file
                        </a>
                      </div>
                    </article>
                  ))}
                </div>
              )}
            </article>

            <article className="panel">
              <SectionIntro
                title="External sources"
                text="Blogs, forums, papers, and self-published proof attempts are tracked separately from internal evidence."
                action={
                  <ShowMoreButton
                    total={filteredSources.length}
                    visible={visible.sources}
                    label="sources"
                    onClick={() => reveal("sources", 4)}
                  />
                }
              />
              <FilterBar onClear={clearSourceFilters}>
                <FilterField label="Search">
                  <input
                    className="filter-input"
                    value={filters.sourceQuery}
                    onChange={(event) => updateFilter("sourceQuery", event.target.value)}
                    placeholder="title, author, url, fallacy tag..."
                  />
                </FilterField>
                <FilterField label="Review status">
                  <select
                    className="filter-input"
                    value={filters.sourceStatus}
                    onChange={(event) => updateFilter("sourceStatus", event.target.value)}
                  >
                    <option value="all">All</option>
                    {sourceStatusOptions.map((status) => (
                      <option key={status} value={status}>
                        {prettyLabel(status)}
                      </option>
                    ))}
                  </select>
                </FilterField>
              </FilterBar>
              {!hasLoaded ? (
                <section className="loading-banner">Waiting for source registry data...</section>
              ) : filteredSources.length === 0 ? (
                <EmptyState title="No sources match the filters" text="Register a source from Operations to start the review trail." />
              ) : (
                <div className="stack-list">
                  {visibleSources.map((source) => (
                    <article key={source.id} className="list-card">
                      <div className="card-head">
                        <div>
                          <h3>{source.title}</h3>
                          <p>{source.id} | {source.direction_slug}</p>
                        </div>
                        <StatusPill value={source.review_status} />
                      </div>
                      <p>{source.summary || "No review summary yet."}</p>
                      <div className="metric-grid">
                        <div>
                          <span className="metric-label">Type</span>
                          <strong>{prettyLabel(source.source_type)}</strong>
                        </div>
                        <div>
                          <span className="metric-label">Claim</span>
                          <strong>{prettyLabel(source.claim_type)}</strong>
                        </div>
                        <div>
                          <span className="metric-label">Authors</span>
                          <strong>{source.authors || "n/a"}</strong>
                        </div>
                        <div>
                          <span className="metric-label">Map</span>
                          <strong>{prettyLabel(source.map_variant || "unspecified")}</strong>
                        </div>
                      </div>
                      <div className="tag-row">
                        {(source.fallacy_tags || []).length > 0 ? (
                          source.fallacy_tags.map((tag) => (
                            <span key={`${source.id}-${tag}`} className="orbit-pill source-tag">
                              {tag}
                            </span>
                          ))
                        ) : (
                          <span className="meta-line">No fallacy tags yet.</span>
                        )}
                      </div>
                      <div className="rubric-grid">
                        <span className={source.rubric.peer_reviewed ? "rubric-badge yes" : "rubric-badge"}>peer reviewed</span>
                        <span className={source.rubric.acknowledged_errors ? "rubric-badge warn" : "rubric-badge"}>acknowledged errors</span>
                        <span className={source.rubric.defines_map_variant ? "rubric-badge yes" : "rubric-badge"}>map variant</span>
                        <span className={source.rubric.distinguishes_empirical_from_proof ? "rubric-badge yes" : "rubric-badge"}>proof distinction</span>
                        <span className={source.rubric.proves_descent ? "rubric-badge yes" : "rubric-badge"}>descent</span>
                        <span className={source.rubric.proves_cycle_exclusion ? "rubric-badge yes" : "rubric-badge"}>cycle exclusion</span>
                        <span className={source.rubric.uses_statistical_argument ? "rubric-badge warn" : "rubric-badge"}>statistical</span>
                        <span className={source.rubric.validation_backed ? "rubric-badge yes" : "rubric-badge"}>validation</span>
                      </div>
                      <p className="meta-line">{source.url || "No URL stored"}{source.year ? ` | ${source.year}` : ""}</p>
                    </article>
                  ))}
                </div>
              )}
            </article>
          </div>
        </section>
        ) : null}

        {activeTab === "queue" ? (
        <section className="tab-panel">
          <article className="panel subpanel">
            <SectionIntro
              title="Create work"
              text="These actions write directly into the same local lab database the CLI uses."
              action={<span className="filter-count">writes live data</span>}
            />
            {actionState.message ? (
              <div className={actionState.tone === "error" ? "action-status action-status-error" : "action-status action-status-success"}>
                {actionState.message}
              </div>
            ) : null}
            <div className="action-grid">
              <form className="action-card" onSubmit={handleRunSubmit}>
                <h3>Queue run</h3>
                <p>Send a reproducible experiment to the worker queue instead of running it inline.</p>
                <div className="action-fields">
                  <ActionField label="Direction">
                    <select
                      className="filter-input"
                      value={quickForms.runDirection}
                      onChange={(event) => updateQuickForm("runDirection", event.target.value)}
                    >
                      {directionOptions.map((direction) => (
                        <option key={direction.slug} value={direction.slug}>
                          {direction.title}
                        </option>
                      ))}
                    </select>
                  </ActionField>
                  <ActionField label="Name" wide>
                    <input
                      className="filter-input"
                      value={quickForms.runName}
                      onChange={(event) => updateQuickForm("runName", event.target.value)}
                      placeholder="record sweep 1-5000"
                      required
                    />
                  </ActionField>
                  <ActionField label="Start">
                    <input
                      className="filter-input"
                      type="number"
                      min="1"
                      value={quickForms.runStart}
                      onChange={(event) => updateQuickForm("runStart", event.target.value)}
                      required
                    />
                  </ActionField>
                  <ActionField label="End">
                    <input
                      className="filter-input"
                      type="number"
                      min="1"
                      value={quickForms.runEnd}
                      onChange={(event) => updateQuickForm("runEnd", event.target.value)}
                      required
                    />
                  </ActionField>
                  <ActionField label="Kernel">
                    <select
                      className="filter-input"
                      value={quickForms.runKernel}
                      onChange={(event) => updateQuickForm("runKernel", event.target.value)}
                    >
                      {kernelOptions.map((kernel) => (
                        <option key={kernel} value={kernel}>
                          {kernel}
                        </option>
                      ))}
                    </select>
                  </ActionField>
                  <ActionField label="Hardware">
                    <select
                      className="filter-input"
                      value={quickForms.runHardware}
                      onChange={(event) => updateQuickForm("runHardware", event.target.value)}
                    >
                      {hardwareOptions.map((hardware) => (
                        <option key={hardware} value={hardware}>
                          {hardware}
                        </option>
                      ))}
                    </select>
                  </ActionField>
                </div>
                <button className="secondary-button action-button" type="submit" disabled={actionState.pendingKey === "queue run"}>
                  {actionState.pendingKey === "queue run" ? "Queueing..." : "Queue run"}
                </button>
              </form>

              <form className="action-card" onSubmit={handleTaskSubmit}>
                <h3>Create task</h3>
                <p>Push the next research step into the shared queue so it can be claimed and tracked.</p>
                <div className="action-fields">
                  <ActionField label="Direction">
                    <select
                      className="filter-input"
                      value={quickForms.taskDirection}
                      onChange={(event) => updateQuickForm("taskDirection", event.target.value)}
                    >
                      {directionOptions.map((direction) => (
                        <option key={direction.slug} value={direction.slug}>
                          {direction.title}
                        </option>
                      ))}
                    </select>
                  </ActionField>
                  <ActionField label="Kind">
                    <input
                      className="filter-input"
                      value={quickForms.taskKind}
                      onChange={(event) => updateQuickForm("taskKind", event.target.value)}
                    />
                  </ActionField>
                  <ActionField label="Title" wide>
                    <input
                      className="filter-input"
                      value={quickForms.taskTitle}
                      onChange={(event) => updateQuickForm("taskTitle", event.target.value)}
                      placeholder="inspect new residue filter"
                      required
                    />
                  </ActionField>
                  <ActionField label="Description" wide>
                    <textarea
                      className="filter-input filter-textarea"
                      value={quickForms.taskDescription}
                      onChange={(event) => updateQuickForm("taskDescription", event.target.value)}
                      placeholder="What should be checked and what would count as success?"
                      required
                    />
                  </ActionField>
                </div>
                <button className="secondary-button action-button" type="submit" disabled={actionState.pendingKey === "task"}>
                  {actionState.pendingKey === "task" ? "Saving..." : "Create task"}
                </button>
              </form>

              <form className="action-card" onSubmit={handleClaimSubmit}>
                <h3>Create claim</h3>
                <p>Capture a mathematical statement as soon as it becomes concrete enough to track.</p>
                <div className="action-fields">
                  <ActionField label="Direction">
                    <select
                      className="filter-input"
                      value={quickForms.claimDirection}
                      onChange={(event) => updateQuickForm("claimDirection", event.target.value)}
                    >
                      {directionOptions.map((direction) => (
                        <option key={direction.slug} value={direction.slug}>
                          {direction.title}
                        </option>
                      ))}
                    </select>
                  </ActionField>
                  <ActionField label="Title" wide>
                    <input
                      className="filter-input"
                      value={quickForms.claimTitle}
                      onChange={(event) => updateQuickForm("claimTitle", event.target.value)}
                      placeholder="Odd reverse tree filter candidate"
                      required
                    />
                  </ActionField>
                  <ActionField label="Statement" wide>
                    <textarea
                      className="filter-input filter-textarea"
                      value={quickForms.claimStatement}
                      onChange={(event) => updateQuickForm("claimStatement", event.target.value)}
                      placeholder="Write the candidate statement in one precise paragraph."
                      required
                    />
                  </ActionField>
                </div>
                <button className="secondary-button action-button" type="submit" disabled={actionState.pendingKey === "claim"}>
                  {actionState.pendingKey === "claim" ? "Saving..." : "Create claim"}
                </button>
              </form>

              <form className="action-card" onSubmit={handleClaimRunLinkSubmit}>
                <h3>Link claim to run</h3>
                <p>Connect a mathematical statement to a specific run so the inspector can show why a claim is supported, tested, or refuted.</p>
                <div className="action-fields">
                  <ActionField label="Claim" wide>
                    <select
                      className="filter-input"
                      value={quickForms.linkClaimId}
                      onChange={(event) => updateQuickForm("linkClaimId", event.target.value)}
                    >
                      {state.claims.length === 0 ? <option value="">No claims yet</option> : null}
                      {state.claims.map((claim) => (
                        <option key={claim.id} value={claim.id}>
                          {claim.id} - {claim.title}
                        </option>
                      ))}
                    </select>
                  </ActionField>
                  <ActionField label="Relation">
                    <select
                      className="filter-input"
                      value={quickForms.linkRelation}
                      onChange={(event) => updateQuickForm("linkRelation", event.target.value)}
                    >
                      {claimRunRelationOptions.map((relation) => (
                        <option key={relation} value={relation}>
                          {prettyLabel(relation)}
                        </option>
                      ))}
                    </select>
                  </ActionField>
                  <ActionField label="Run" wide>
                    <select
                      className="filter-input"
                      value={quickForms.linkRunId}
                      onChange={(event) => updateQuickForm("linkRunId", event.target.value)}
                    >
                      {state.runs.length === 0 ? <option value="">No runs yet</option> : null}
                      {state.runs.map((run) => (
                        <option key={run.id} value={run.id}>
                          {run.id} - {run.name} ({run.status})
                        </option>
                      ))}
                    </select>
                  </ActionField>
                </div>
                <button className="secondary-button action-button" type="submit" disabled={actionState.pendingKey === "claim link"}>
                  {actionState.pendingKey === "claim link" ? "Linking..." : "Save claim link"}
                </button>
              </form>

              <form className="action-card" onSubmit={handleSourceSubmit}>
                <h3>Register source</h3>
                <p>Store an external proof attempt, blog post, paper, or forum thread before trusting or rejecting it.</p>
                <div className="action-fields">
                  <ActionField label="Direction">
                    <select
                      className="filter-input"
                      value={quickForms.sourceDirection}
                      onChange={(event) => updateQuickForm("sourceDirection", event.target.value)}
                    >
                      {directionOptions.map((direction) => (
                        <option key={direction.slug} value={direction.slug}>
                          {direction.title}
                        </option>
                      ))}
                    </select>
                  </ActionField>
                  <ActionField label="Source type">
                    <select
                      className="filter-input"
                      value={quickForms.sourceType}
                      onChange={(event) => updateQuickForm("sourceType", event.target.value)}
                    >
                      {sourceTypeOptions.map((option) => (
                        <option key={option} value={option}>
                          {prettyLabel(option)}
                        </option>
                      ))}
                    </select>
                  </ActionField>
                  <ActionField label="Claim type">
                    <select
                      className="filter-input"
                      value={quickForms.sourceClaimType}
                      onChange={(event) => updateQuickForm("sourceClaimType", event.target.value)}
                    >
                      {sourceClaimTypeOptions.map((option) => (
                        <option key={option} value={option}>
                          {prettyLabel(option)}
                        </option>
                      ))}
                    </select>
                  </ActionField>
                  <ActionField label="Map variant">
                    <select
                      className="filter-input"
                      value={quickForms.sourceMapVariant}
                      onChange={(event) => updateQuickForm("sourceMapVariant", event.target.value)}
                    >
                      {mapVariantOptions.map((option) => (
                        <option key={option} value={option}>
                          {prettyLabel(option)}
                        </option>
                      ))}
                    </select>
                  </ActionField>
                  <ActionField label="Year">
                    <input
                      className="filter-input"
                      value={quickForms.sourceYear}
                      onChange={(event) => updateQuickForm("sourceYear", event.target.value)}
                      placeholder="2026"
                    />
                  </ActionField>
                  <ActionField label="Title" wide>
                    <input
                      className="filter-input"
                      value={quickForms.sourceTitle}
                      onChange={(event) => updateQuickForm("sourceTitle", event.target.value)}
                      placeholder="Personal proof attempt by descent and cycle exclusion"
                      required
                    />
                  </ActionField>
                  <ActionField label="Authors" wide>
                    <input
                      className="filter-input"
                      value={quickForms.sourceAuthors}
                      onChange={(event) => updateQuickForm("sourceAuthors", event.target.value)}
                      placeholder="Author names"
                    />
                  </ActionField>
                  <ActionField label="URL" wide>
                    <input
                      className="filter-input"
                      value={quickForms.sourceUrl}
                      onChange={(event) => updateQuickForm("sourceUrl", event.target.value)}
                      placeholder="https://..."
                    />
                  </ActionField>
                  <ActionField label="Summary" wide>
                    <textarea
                      className="filter-input filter-textarea"
                      value={quickForms.sourceSummary}
                      onChange={(event) => updateQuickForm("sourceSummary", event.target.value)}
                      placeholder="What does the source claim, in one precise paragraph?"
                      required
                    />
                  </ActionField>
                  <ActionField label="Initial fallacy tags" wide>
                    <input
                      className="filter-input"
                      value={quickForms.sourceTags}
                      onChange={(event) => updateQuickForm("sourceTags", event.target.value)}
                      placeholder="empirical-not-proof, almost-all-not-all"
                    />
                  </ActionField>
                  <div className="tag-catalog action-field-wide">
                    <span>Known fallacy tags</span>
                    <div className="tag-catalog-grid">
                      {fallacyCatalog.map((item) => (
                        <button
                          key={item.tag}
                          type="button"
                          className="orbit-pill source-tag tag-catalog-button"
                          onClick={() => addFallacyTag("sourceTags", item.tag)}
                          title={item.description}
                        >
                          {item.tag}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
                <button className="secondary-button action-button" type="submit" disabled={actionState.pendingKey === "source"}>
                  {actionState.pendingKey === "source" ? "Saving..." : "Register source"}
                </button>
              </form>

              <form className="action-card" onSubmit={handleReviewSubmit}>
                <h3>Review direction</h3>
                <p>Recompute the current score and status from the evidence already linked in the lab.</p>
                <div className="action-fields">
                  <ActionField label="Direction" wide>
                    <select
                      className="filter-input"
                      value={quickForms.reviewDirection}
                      onChange={(event) => updateQuickForm("reviewDirection", event.target.value)}
                    >
                      {directionOptions.map((direction) => (
                        <option key={direction.slug} value={direction.slug}>
                          {direction.title}
                        </option>
                      ))}
                    </select>
                  </ActionField>
                </div>
                {reviewResult ? (
                  <div className="review-card">
                    <div className="card-head">
                      <strong>{reviewResult.direction.title}</strong>
                      <StatusPill value={reviewResult.direction.status} />
                    </div>
                    <p>{reviewResult.rationale}</p>
                    <p className="meta-line">
                      validated {reviewResult.validated_runs} | supported {reviewResult.supported_claims} | refuted {reviewResult.refuted_claims}
                    </p>
                  </div>
                ) : null}
                <button className="secondary-button action-button" type="submit" disabled={actionState.pendingKey === "direction review"}>
                  {actionState.pendingKey === "direction review" ? "Reviewing..." : "Review direction"}
                </button>
              </form>

              <form className="action-card" onSubmit={handleSourceReviewSubmit}>
                <h3>Review source</h3>
                <p>Move a source from intake toward flagged, supported, or refuted only after an explicit rubric pass.</p>
                <div className="action-fields">
                  <ActionField label="Source" wide>
                    <select
                      className="filter-input"
                      value={quickForms.sourceReviewId}
                      onChange={(event) => selectSourceForReview(event.target.value)}
                    >
                      {state.sources.length === 0 ? <option value="">No sources yet</option> : null}
                      {state.sources.map((source) => (
                        <option key={source.id} value={source.id}>
                          {source.id} - {source.title}
                        </option>
                      ))}
                    </select>
                  </ActionField>
                  {selectedReviewSource ? (
                    <div className="review-card action-field-wide">
                      <div className="card-head">
                        <strong>{selectedReviewSource.title}</strong>
                        <StatusPill value={selectedReviewSource.review_status} />
                      </div>
                      <p>{selectedReviewSource.summary || "No summary stored yet."}</p>
                      <p className="meta-line">
                        {prettyLabel(selectedReviewSource.source_type)} | {prettyLabel(selectedReviewSource.claim_type)} | {prettyLabel(selectedReviewSource.map_variant)}
                      </p>
                    </div>
                  ) : null}
                  <ActionField label="Status">
                    <select
                      className="filter-input"
                      value={quickForms.sourceReviewStatus}
                      onChange={(event) => updateQuickForm("sourceReviewStatus", event.target.value)}
                    >
                      {sourceStatusOptions.map((status) => (
                        <option key={status} value={status}>
                          {prettyLabel(status)}
                        </option>
                      ))}
                    </select>
                  </ActionField>
                  <ActionField label="Map variant">
                    <select
                      className="filter-input"
                      value={quickForms.sourceReviewMapVariant}
                      onChange={(event) => updateQuickForm("sourceReviewMapVariant", event.target.value)}
                    >
                      {mapVariantOptions.map((option) => (
                        <option key={option} value={option}>
                          {prettyLabel(option)}
                        </option>
                      ))}
                    </select>
                  </ActionField>
                  <ActionField label="Fallacy tags">
                    <input
                      className="filter-input"
                      value={quickForms.sourceReviewTags}
                      onChange={(event) => updateQuickForm("sourceReviewTags", event.target.value)}
                      placeholder="unchecked-generalization, reverse-tree-gap"
                    />
                  </ActionField>
                  <ActionField label="Review notes" wide>
                    <textarea
                      className="filter-input filter-textarea"
                      value={quickForms.sourceReviewNotes}
                      onChange={(event) => updateQuickForm("sourceReviewNotes", event.target.value)}
                      placeholder="What is missing, what is legitimate, and what still needs falsification?"
                    />
                  </ActionField>
                  <div className="tag-catalog action-field-wide">
                    <span>Known fallacy tags</span>
                    <div className="tag-catalog-grid">
                      {fallacyCatalog.map((item) => (
                        <button
                          key={`review-${item.tag}`}
                          type="button"
                          className="orbit-pill source-tag tag-catalog-button"
                          onClick={() => addFallacyTag("sourceReviewTags", item.tag)}
                          title={item.description}
                        >
                          {item.tag}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="action-field action-field-wide">
                    <span>Review rubric</span>
                    <div className="rubric-select-grid">
                      {rubricFieldOptions.map((field) => {
                        const formKey = {
                          peer_reviewed: "sourceReviewPeerReviewed",
                          acknowledged_errors: "sourceReviewAcknowledgedErrors",
                          defines_map_variant: "sourceReviewDefinesMapVariant",
                          distinguishes_empirical_from_proof: "sourceReviewDistinguishesProof",
                          proves_descent: "sourceReviewProvesDescent",
                          proves_cycle_exclusion: "sourceReviewProvesCycleExclusion",
                          uses_statistical_argument: "sourceReviewUsesStatisticalArgument",
                          validation_backed: "sourceReviewValidationBacked"
                        }[field.key];
                        return (
                          <label key={field.key} className="rubric-control">
                            <span>{field.label}</span>
                            <select
                              className="filter-input"
                              value={quickForms[formKey]}
                              onChange={(event) => updateQuickForm(formKey, event.target.value)}
                            >
                              <option value="unknown">Unknown</option>
                              <option value="yes">Yes</option>
                              <option value="no">No</option>
                            </select>
                          </label>
                        );
                      })}
                    </div>
                  </div>
                </div>
                {sourceReviewResult ? (
                  <div className="review-card">
                    <div className="card-head">
                      <strong>{sourceReviewResult.title}</strong>
                      <StatusPill value={sourceReviewResult.review_status} />
                    </div>
                    <p>{sourceReviewResult.summary || "Review saved."}</p>
                    <p className="meta-line">
                      {(sourceReviewResult.fallacy_tags || []).length > 0
                        ? sourceReviewResult.fallacy_tags.join(", ")
                        : "No fallacy tags recorded"}
                    </p>
                  </div>
                ) : null}
                <button className="secondary-button action-button" type="submit" disabled={actionState.pendingKey === "source review"}>
                  {actionState.pendingKey === "source review" ? "Saving..." : "Review source"}
                </button>
              </form>

              <form className="action-card" onSubmit={handleProbeSubmit}>
                <h3>Run modular probe</h3>
                <p>Quickly falsify residue-class claims before they contaminate a proof attempt.</p>
                <div className="action-fields">
                  <ActionField label="Modulus">
                    <input
                      className="filter-input"
                      type="number"
                      min="2"
                      value={quickForms.probeModulus}
                      onChange={(event) => updateQuickForm("probeModulus", event.target.value)}
                      required
                    />
                  </ActionField>
                  <ActionField label="Allowed residues">
                    <input
                      className="filter-input"
                      value={quickForms.probeResidues}
                      onChange={(event) => updateQuickForm("probeResidues", event.target.value)}
                      placeholder="5"
                      required
                    />
                  </ActionField>
                  <ActionField label="Search limit" wide>
                    <input
                      className="filter-input"
                      type="number"
                      min="3"
                      value={quickForms.probeLimit}
                      onChange={(event) => updateQuickForm("probeLimit", event.target.value)}
                      required
                    />
                  </ActionField>
                </div>
                {probeResult ? (
                  <div className="review-card">
                    <div className="card-head">
                      <strong>mod {probeResult.modulus}</strong>
                      <span className="filter-count">
                        {probeResult.first_counterexample ? "counterexample found" : "no counterexample found"}
                      </span>
                    </div>
                    <p>{probeResult.rationale}</p>
                    <p className="meta-line">
                      checked {probeResult.checked_odd_values} odd seeds up to {probeResult.checked_limit}
                    </p>
                  </div>
                ) : null}
                <button className="secondary-button action-button" type="submit" disabled={actionState.pendingKey === "modular probe"}>
                  {actionState.pendingKey === "modular probe" ? "Probing..." : "Run probe"}
                </button>
              </form>
            </div>
          </article>
          <SectionIntro
            title="What should happen next"
            text="Use this part of the dashboard when you want to know the next human-scale research actions."
            action={
              <ShowMoreButton
                total={filteredTasks.length}
                visible={visible.tasks}
                label="tasks"
                onClick={() => reveal("tasks", 6)}
              />
            }
          />
          <article className="panel subpanel">
            <SectionIntro
              title="Worker dispatch"
              text="Queued runs are real records in SQLite. They move only when a compatible worker claims them."
              action={<span className="filter-count">{liveWorkers.length} workers visible</span>}
            />
            {state.workers.length === 0 ? (
              <EmptyState
                title="No worker registered"
                text="Use `npm run stack:start:worker` or `python -m collatz_lab.cli worker start --hardware auto`."
              />
            ) : (
              <div className="stack-list">
                {state.workers.slice(0, 4).map((worker) => (
                  <article key={worker.id} className="list-card">
                    <div className="card-head">
                      <div>
                        <h3>{worker.name}</h3>
                        <p>{worker.id}</p>
                      </div>
                      <StatusPill value={worker.status} />
                    </div>
                    <p className="meta-line">
                      {worker.hardware} | current run {worker.current_run_id ?? "none"}
                    </p>
                    <p className="meta-line">heartbeat {worker.last_heartbeat_at ?? "unknown"}</p>
                  </article>
                ))}
              </div>
            )}
          </article>
          <article className="panel subpanel">
            <SectionIntro
              title="Task filters"
              text="Search by task title, direction, owner, kind, or status."
              action={<span className="filter-count">{filteredTasks.length} matches</span>}
            />
            <FilterBar onClear={clearTaskFilters}>
              <FilterField label="Search">
                <input
                  className="filter-input"
                  value={filters.taskQuery}
                  onChange={(event) => updateFilter("taskQuery", event.target.value)}
                  placeholder="task, direction, owner..."
                />
              </FilterField>
              <FilterField label="Status">
                <select
                  className="filter-input"
                  value={filters.taskStatus}
                  onChange={(event) => updateFilter("taskStatus", event.target.value)}
                >
                  <option value="all">All</option>
                  <option value="open">Open</option>
                  <option value="in_progress">In progress</option>
                  <option value="done">Done</option>
                  <option value="frozen">Frozen</option>
                </select>
              </FilterField>
            </FilterBar>
          </article>
          {!hasLoaded ? (
            <section className="loading-banner">Waiting for real task data...</section>
          ) : filteredTasks.length === 0 ? (
            <EmptyState title="Queue is empty for this filter" text="Create tasks from the CLI or API and they will appear here." />
          ) : (
            <div className="stack-list">
              {visibleTasks.map((task) => (
                <article key={task.id} className="panel queue-card">
                  <div className="card-head">
                    <div>
                      <h3>{task.title}</h3>
                      <p>{task.id} | {task.direction_slug}</p>
                    </div>
                    <StatusPill value={task.status} />
                  </div>
                  <p>{task.description}</p>
                  <div className="metric-grid three-up">
                    <div>
                      <span className="metric-label">Owner</span>
                      <strong>{task.owner}</strong>
                    </div>
                    <div>
                      <span className="metric-label">Kind</span>
                      <strong>{task.kind}</strong>
                    </div>
                    <div>
                      <span className="metric-label">Priority</span>
                      <strong>{task.priority}</strong>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>
        ) : null}

        {activeTab === "guide" ? (
        <section className="tab-panel">
          <SectionIntro
            title="Guide"
            text="This page explains what each role does and how results should move through the lab."
          />
          <div className="guide-grid">
            <article className="panel">
              <h3>Agent workflow</h3>
              <p className="guide-lead">
                Each role is intentionally small. The important part is the chain from idea to evidence to review, not the number of boxes on screen.
              </p>
              <ol className="timeline">
                <li>
                  <strong>compute-agent</strong>
                  <p>Runs interval sweeps, saves metrics, and produces reproducible artifacts.</p>
                </li>
                <li>
                  <strong>theory-agent</strong>
                  <p>Proposes claims, filters, inverse-tree ideas, and candidate invariants.</p>
                </li>
                <li>
                  <strong>validator-agent</strong>
                  <p>Replays results with independent logic before a direction or claim is promoted.</p>
                </li>
                <li>
                  <strong>integrator</strong>
                  <p>Reviews evidence, links runs to claims, and decides whether work stays active, promising, frozen, or refuted.</p>
                </li>
              </ol>
              <div className="workflow-footer">
                <span>Promotion rule</span>
                <p>Nothing moves to `main` without a generated report and a validation path.</p>
              </div>
            </article>

            <article className="panel">
              <h3>What to do next</h3>
              <div className="stack-list compact-stack">
                <article className="list-card">
                  <strong>1. Start in Overview</strong>
                  <p>Check validated runs, supported claims, and direction status.</p>
                </article>
                <article className="list-card">
                  <strong>2. Open Evidence</strong>
                  <p>Inspect actual run summaries and artifact links before trusting a pattern.</p>
                </article>
                <article className="list-card">
                  <strong>3. Open Queue</strong>
                  <p>Pick the next task only after you understand the current evidence.</p>
                </article>
                <article className="list-card">
                  <strong>4. Read the roadmap</strong>
                  <p>The phase plan is stored in <code>research/ROADMAP.md</code>.</p>
                </article>
              </div>
            </article>

            <article className="panel">
              <h3>Source review rubric</h3>
              <div className="stack-list compact-stack">
                <article className="list-card">
                  <strong>1. Classify the source</strong>
                  <p>Mark it as peer-reviewed, preprint, self-published, forum, blog, Q&A, or news.</p>
                </article>
                <article className="list-card">
                  <strong>2. Separate proof from evidence</strong>
                  <p>Computational checks and almost-all results never count as a full proof on their own.</p>
                </article>
                <article className="list-card">
                  <strong>3. Require map discipline</strong>
                  <p>Any source that blurs standard, shortcut, and odd-only variants should be flagged.</p>
                </article>
                <article className="list-card">
                  <strong>4. Falsify local lemmas first</strong>
                  <p>Run modular probes before spending serious attention on a proof attempt.</p>
                </article>
              </div>
            </article>

            <article className="panel">
              <h3>Consensus baseline</h3>
              {state.baseline ? (
                <div className="stack-list compact-stack">
                  <article className="list-card">
                    <strong>{prettyLabel(state.baseline.problem_status)}</strong>
                    <p>{state.baseline.note}</p>
                  </article>
                  {state.baseline.items.map((item) => (
                    <article key={item.title} className="list-card">
                      <strong>{item.title}</strong>
                      <p>{item.detail}</p>
                    </article>
                  ))}
                </div>
              ) : (
                <EmptyState title="No baseline payload" text="The API baseline endpoint has not responded yet." />
              )}
            </article>
          </div>
        </section>
        ) : null}
        </div>
        <RedditIntelRail
          feed={state.redditFeed}
          onImportPost={handleRedditImport}
          pendingKey={actionState.pendingKey}
        />
        </div>
      </div>
    </main>
  );
}
