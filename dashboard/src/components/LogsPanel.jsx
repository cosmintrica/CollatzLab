import { useState, useEffect, useRef, useCallback, memo } from "react";
import { endpoints } from "../config.js";

const LEVELS = ["", "INFO", "WARNING", "ERROR"];
const LEVEL_LABELS = { "": "All", INFO: "Info", WARNING: "Warn", ERROR: "Error" };
const LEVEL_CLASS = { INFO: "log-info", WARNING: "log-warn", ERROR: "log-error", CRITICAL: "log-error", DEBUG: "log-debug" };

function LogRow({ entry }) {
  const [expanded, setExpanded] = useState(false);
  const multiline = entry.msg.includes("\n");
  const firstLine = multiline ? entry.msg.split("\n")[0] : entry.msg;
  const tsShort = (entry.ts || "").slice(0, 19);

  return (
    <div
      className={`log-row ${LEVEL_CLASS[entry.level] || ""} ${entry.kind === "run-failure" ? "log-run-fail" : ""} ${entry.kind === "run-superseded" ? "log-run-superseded" : ""}`}
      onClick={multiline ? () => setExpanded(!expanded) : undefined}
      style={multiline ? { cursor: "pointer" } : undefined}
    >
      <span className="log-ts">{tsShort}</span>
      <span className={`log-level log-level-${entry.level?.toLowerCase()}`}>{entry.level}</span>
      <span className="log-source">{entry.source}</span>
      <span className="log-msg">{expanded ? entry.msg : firstLine}{multiline && !expanded ? " ..." : ""}</span>
    </div>
  );
}

export default memo(function LogsPanel({ open, onClose }) {
  const [entries, setEntries] = useState([]);
  const [search, setSearch] = useState("");
  const [level, setLevel] = useState("");
  const [source, setSource] = useState("");
  const [loading, setLoading] = useState(false);
  const [sortAsc, setSortAsc] = useState(false);
  const listRef = useRef(null);
  const panelRef = useRef(null);

  // Close on click outside
  useEffect(() => {
    if (!open) return;
    const handler = (e) => {
      if (panelRef.current && !panelRef.current.contains(e.target)
          && !e.target.closest(".logs-toggle-btn")) onClose();
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open, onClose]);

  const fetchLogs = useCallback(() => {
    if (!open) return;
    setLoading(true);
    const params = new URLSearchParams();
    if (search) params.set("q", search);
    if (level) params.set("level", level);
    if (source) params.set("source", source);
    params.set("limit", "500");
    fetch(`${endpoints.logs}?${params}`)
      .then((r) => (r.ok ? r.json() : []))
      .then((data) => { setEntries(data); setLoading(false); })
      .catch(() => setLoading(false));
  }, [open, search, level, source]);

  useEffect(() => { fetchLogs(); }, [fetchLogs]);

  // Auto-refresh every 10s when open
  useEffect(() => {
    if (!open) return;
    const id = setInterval(fetchLogs, 10000);
    return () => clearInterval(id);
  }, [open, fetchLogs]);

  if (!open) return null;

  const sorted = sortAsc ? [...entries].reverse() : entries;

  // Extract unique sources for filter
  const sources = [...new Set(entries.map((e) => e.source).filter(Boolean))].sort();

  return (
    <div className="logs-panel" ref={panelRef}>
      <div className="logs-header">
        <div className="logs-title-row">
          <span className="logs-title">System logs</span>
          <span className="logs-count">{entries.length}</span>
          <button type="button" className="logs-refresh" onClick={fetchLogs} title="Refresh">
            {loading ? "..." : "\u21BB"}
          </button>
        </div>
        <div className="logs-filters">
          <input
            className="logs-search"
            type="text"
            placeholder="Search logs..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <select className="logs-select" value={level} onChange={(e) => setLevel(e.target.value)}>
            {LEVELS.map((l) => (
              <option key={l} value={l}>{LEVEL_LABELS[l]}</option>
            ))}
          </select>
          <select className="logs-select" value={source} onChange={(e) => setSource(e.target.value)}>
            <option value="">All sources</option>
            {sources.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
          <button
            type="button"
            className="logs-sort-btn"
            onClick={() => setSortAsc(!sortAsc)}
            title={sortAsc ? "Oldest first" : "Newest first"}
          >
            {sortAsc ? "\u2191 Old" : "\u2193 New"}
          </button>
        </div>
        <button type="button" className="logs-close" onClick={onClose} title="Close">{"\u2715"}</button>
      </div>
      <div className="logs-list" ref={listRef}>
        {sorted.length === 0 ? (
          <div className="logs-empty">{loading ? "Loading..." : "No log entries match."}</div>
        ) : (
          sorted.map((entry, i) => <LogRow key={`${entry.ts}-${i}`} entry={entry} />)
        )}
      </div>
    </div>
  );
})
