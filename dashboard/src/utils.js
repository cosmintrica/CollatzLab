export function fileSafeLabel(value) {
  return String(value ?? "evidence")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "") || "evidence";
}

export function downloadText(filename, content, mimeType = "text/plain;charset=utf-8") {
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

export function downloadJson(filename, payload) {
  downloadText(filename, JSON.stringify(payload, null, 2), "application/json;charset=utf-8");
}

export function triggerDownload(filename, content, type) {
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

export function exportJsonFile(filename, payload) {
  triggerDownload(filename, `${JSON.stringify(payload, null, 2)}\n`, "application/json");
}

export function exportTextFile(filename, payload) {
  triggerDownload(filename, payload, "text/plain;charset=utf-8");
}

export function asList(payload, keys) {
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

export function prettyLabel(value) {
  return String(value).replaceAll("-", " ").replaceAll("_", " ");
}

export function normalize(value) {
  return String(value ?? "").trim().toLowerCase();
}

export function timestampValue(value) {
  if (!value) {
    return 0;
  }
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

export function formatTimestamp(value) {
  if (!value) {
    return "No timestamp";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  const dd = String(parsed.getDate()).padStart(2, "0");
  const mm = String(parsed.getMonth() + 1).padStart(2, "0");
  const yyyy = parsed.getFullYear();
  const hh = String(parsed.getHours()).padStart(2, "0");
  const min = String(parsed.getMinutes()).padStart(2, "0");
  const ss = String(parsed.getSeconds()).padStart(2, "0");
  return `${dd}/${mm}/${yyyy}, ${hh}:${min}:${ss}`;
}

export function formatDate(value) {
  if (!value) return "–";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  const dd = String(parsed.getDate()).padStart(2, "0");
  const mm = String(parsed.getMonth() + 1).padStart(2, "0");
  const yyyy = parsed.getFullYear();
  return `${dd}/${mm}/${yyyy}`;
}

const MONTH_SHORT = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
export function formatCompactTimestamp(value) {
  if (!value) return "";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return "";
  const dd = d.getDate();
  const mon = MONTH_SHORT[d.getMonth()];
  const yyyy = d.getFullYear();
  const hh = String(d.getHours()).padStart(2, "0");
  const min = String(d.getMinutes()).padStart(2, "0");
  return `${dd} ${mon} ${yyyy} ${hh}:${min}`;
}

export function formatRelativeTime(value) {
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

export function latestTimestamp(...values) {
  return [...values]
    .filter(Boolean)
    .sort((left, right) => timestampValue(right) - timestampValue(left))[0] || "";
}

export function artifactLabel(path, fallback) {
  if (!path) {
    return fallback;
  }
  const pieces = String(path).split(/[\\/]/);
  return pieces[pieces.length - 1] || fallback;
}

export function includesQuery(values, query) {
  const needle = normalize(query);
  if (!needle) {
    return true;
  }
  return values.some((value) => normalize(value).includes(needle));
}

export function parseCsvList(value) {
  return String(value ?? "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

export function runRangeSize(run) {
  if (!run) {
    return 0;
  }
  return Math.max(0, (Number(run.range_end) || 0) - (Number(run.range_start) || 0) + 1);
}

export function approximatePowerOfTwo(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return "";
  }
  return `~2^${Math.log2(numeric).toFixed(2)}`;
}

export function approximatePowerOfTwoLatex(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return "";
  }
  return `{\\approx}2^{${Math.log2(numeric).toFixed(2)}}`;
}

export function powerGapToMilestone(value, milestoneExponent = 71) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return "";
  }
  const gap = milestoneExponent - Math.log2(numeric);
  return gap >= 0
    ? `${gap.toFixed(2)} below 2^${milestoneExponent}`
    : `${Math.abs(gap).toFixed(2)} above 2^${milestoneExponent}`;
}

export function runRangeMagnitudeLatex(run) {
  if (!run) return "";
  const s = approximatePowerOfTwoLatex(run.range_start);
  const e = approximatePowerOfTwoLatex(run.range_end);
  if (!s || !e) return "";
  return `${s} \\to ${e}`;
}

export function runMilestoneLabelLatex(run, milestoneExponent = 71) {
  if (!run) return "";
  const topValue = Math.max(
    Number(run.range_end) || 0,
    Number(run.checkpoint?.last_processed) || 0,
    Number(run.metrics?.last_processed) || 0
  );
  if (topValue <= 0) return "";
  const exp = Math.log2(topValue).toFixed(2);
  const gap = milestoneExponent - Math.log2(topValue);
  const gapStr = Math.abs(gap).toFixed(2);
  const gapLatex = gap >= 0
    ? `${gapStr}\\text{ below }2^{${milestoneExponent}}`
    : `${gapStr}\\text{ above }2^{${milestoneExponent}}`;
  return `{\\approx}2^{${exp}} \\;\\mid\\; ${gapLatex}`;
}

export function runRangeMagnitude(run) {
  if (!run) {
    return "";
  }
  const startLabel = approximatePowerOfTwo(run.range_start);
  const endLabel = approximatePowerOfTwo(run.range_end);
  if (!startLabel || !endLabel) {
    return "";
  }
  return `${startLabel} -> ${endLabel}`;
}

export function runMilestoneLabel(run) {
  if (!run) {
    return "";
  }
  const topValue = Math.max(
    Number(run.range_end) || 0,
    Number(run.checkpoint?.last_processed) || 0,
    Number(run.metrics?.last_processed) || 0
  );
  if (topValue <= 0) {
    return "";
  }
  const powerLabel = approximatePowerOfTwo(topValue);
  const gapLabel = powerGapToMilestone(topValue, 71);
  return `${powerLabel} | ${gapLabel}`;
}

export function extractFailureReason(summary) {
  const text = String(summary || "").trim();
  if (!text) {
    return "";
  }
  const marker = text.toLowerCase().indexOf(" failed: ");
  if (marker >= 0) {
    return text.slice(marker + 9).trim();
  }
  return text;
}

export function describeRunPurpose(run) {
  if (!run) {
    return "No run selected.";
  }
  const text = `${run.name || ""} ${run.summary || ""}`.toLowerCase();
  if ((run.name || "").startsWith("recover-prefix-")) {
    return "Recovered exact prefix preserved from a run that hit the signed-64-bit frontier.";
  }
  if ((run.name || "").startsWith("recover-tail-")) {
    return "Exact overflow-safe CPU recovery for the uncovered tail after a signed-64-bit frontier failure.";
  }
  if (run.direction_slug === "verification") {
    if (text.includes("continuous") && run.hardware === "gpu") {
      return "Continuous GPU verification stream for large-interval record search, verification throughput, and fresh live checkpoints.";
    }
    if (text.includes("continuous") && run.hardware === "cpu") {
      return "Continuous CPU verification stream for bounded interval evidence, replay coverage, and record tracking.";
    }
    if (run.status === "validated") {
      return "Bounded verification run that already passed an independent replay.";
    }
    return "Verification sweep used for record tracking, replay, and later validation.";
  }
  if (run.direction_slug === "inverse-tree-parity") {
    return "Structure-facing run used to test parity or residue ideas against actual intervals.";
  }
  if (run.direction_slug === "two-adic-v2") {
    return "2-adic / odd-step probe used to study v2(3n+1) and compressed odd dynamics.";
  }
  if (run.direction_slug === "lemma-workspace") {
    return "Claim-supporting run linked back into the proof-facing workspace.";
  }
  return "General Collatz compute record.";
}

export function describeRunStatusDetail(run) {
  if (!run) {
    return "";
  }
  if (run.status === "failed") {
    return extractFailureReason(run.summary);
  }
  if ((run.name || "").startsWith("recover-tail-")) {
    return "Exact CPU fallback is slower here because it avoids the signed-64-bit overflow frontier.";
  }
  return "";
}

export function classifyRunCategory(run) {
  if (!run) {
    return "standard";
  }
  const summary = String(run.summary || "").toLowerCase();
  const name = String(run.name || "").toLowerCase();
  if (name.startsWith("recover-prefix-") || name.startsWith("recover-tail-")) {
    return "recovery";
  }
  if (summary.includes("superseded by col-")) {
    return "legacy-superseded";
  }
  if (summary.startsWith("validation failed:")) {
    return "legacy-failed";
  }
  if (name.includes("continuous")) {
    return "continuous";
  }
  if (
    run.direction_slug === "two-adic-v2" ||
    run.direction_slug === "hypothesis-sandbox" ||
    run.kernel === "cpu-barina"
  ) {
    return "experimental";
  }
  return "standard";
}

export function parseClaimNotes(notes) {
  const raw = String(notes || "").trim();
  if (!raw) {
    return { snapshot: null, history: [], raw: "" };
  }

  const managedMatch = raw.match(/<!-- auto-consolidation:start -->([\s\S]*?)<!-- auto-consolidation:end -->/);
  const managedText = managedMatch?.[1]?.trim() || "";
  const readManagedValue = (label) => {
    const pattern = new RegExp(`- ${label}: ([^\\n]+)`);
    return managedText.match(pattern)?.[1]?.trim() || "";
  };
  const runCoverage = readManagedValue("Run coverage");
  const snapshot = managedText
    ? {
        updatedAt: readManagedValue("Last updated at"),
        sourceTask: readManagedValue("Last source task"),
        entryCount: readManagedValue("Consolidated record entries"),
        uniqueSeeds: readManagedValue("Unique record seeds"),
        linkedRuns: readManagedValue("Supporting runs currently linked"),
        runCoveragePreview: runCoverage
          ? runCoverage.split(",").map((item) => item.trim()).filter(Boolean).slice(0, 8)
          : []
      }
    : null;

  const history = [...raw.matchAll(
    /## Task execution update ([^\n]+)\s+[\s\S]*?- Source task: ([^\n]+)\s+[\s\S]*?- Consolidated record entries: ([^\n]+)\s+[\s\S]*?- Unique record seeds: ([^\n]+)\s+[\s\S]*?- Supporting runs linked in this pass: ([^\n]+)/g
  )].map((match) => ({
    updatedAt: match[1].trim(),
    sourceTask: match[2].trim(),
    entryCount: match[3].trim(),
    uniqueSeeds: match[4].trim(),
    linkedRuns: match[5].trim()
  }));

  return { snapshot, history, raw };
}

export function appendCsvTag(value, tag) {
  const next = parseCsvList(value);
  if (!next.includes(tag)) {
    next.push(tag);
  }
  return next.join(", ");
}

export function rubricValueToSelect(value) {
  if (value === true) {
    return "yes";
  }
  if (value === false) {
    return "no";
  }
  return "unknown";
}

export function selectToRubricValue(value) {
  if (value === "yes") {
    return true;
  }
  if (value === "no") {
    return false;
  }
  return null;
}

export function countBy(items, getKey) {
  return items.reduce((accumulator, item) => {
    const key = getKey(item) || "unknown";
    accumulator[key] = (accumulator[key] || 0) + 1;
    return accumulator;
  }, {});
}

export function taskIntent(task) {
  const text = `${task?.title || ""} ${task?.description || ""}`.toLowerCase();
  if (
    task?.kind === "experiment" ||
    task?.direction_slug === "verification" ||
    ["sweep", "queue", "run", "kernel", "checkpoint"].some((keyword) => text.includes(keyword))
  ) {
    return "compute";
  }
  return "theory";
}

export function formatMathNum(n) {
  const num = typeof n === "bigint" ? Number(n) : Number(n);
  if (num === 0) return { m: "0", e: null };
  const abs = Math.abs(num);
  if (abs < 100000) return { m: num.toLocaleString(), e: null };
  const e = Math.floor(Math.log10(abs));
  const mantissa = num / 10 ** e;
  const mStr = mantissa === Math.floor(mantissa) ? String(Math.floor(mantissa)) : mantissa.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
  return { m: mStr, e };
}
