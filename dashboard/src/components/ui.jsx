import { useEffect, useRef } from "react";
import { prettyLabel, formatMathNum } from "../utils.js";
import katex from "katex";
import "katex/dist/katex.min.css";

/**
 * Renders a LaTeX expression using KaTeX.
 * Pass display=true for block/display mode, false for inline.
 * className is applied to the wrapper div/span.
 */
export function KatexExpr({ latex, display = false, className = "" }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current || !latex) return;
    try {
      katex.render(latex, ref.current, {
        displayMode: display,
        throwOnError: false,
        trust: false,
      });
    } catch (_) {
      if (ref.current) ref.current.textContent = latex;
    }
  }, [latex, display]);
  return display
    ? <div ref={ref} className={`katex-block ${className}`} />
    : <span ref={ref} className={`katex-inline ${className}`} />;
}

export function StatusPill({ value }) {
  return <span className={`status-pill status-${value}`}>{prettyLabel(value)}</span>;
}

export function SummaryCard({ label, value, note }) {
  return (
    <article className="summary-card">
      <span className="summary-label">{label}</span>
      <strong>{value}</strong>
      <p>{note}</p>
    </article>
  );
}

export function SectionIntro({ title, text, action }) {
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

export function MathNum({ value }) {
  const { m, e } = formatMathNum(value);
  if (e === null) return <>{m}</>;
  return <>{m}{"\u00b7"}10<sup>{e}</sup></>;
}

export function Legend() {
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

export function EmptyState({ title, text }) {
  return (
    <article className="empty-state">
      <h3>{title}</h3>
      <p>{text}</p>
    </article>
  );
}

export function ShowMoreButton({ total, visible, onClick, label }) {
  if (total <= visible) {
    return null;
  }
  return (
    <button className="secondary-button" type="button" onClick={onClick}>
      Show more {label} ({total - visible} hidden)
    </button>
  );
}

export function FilterField({ label, children }) {
  return (
    <label className="filter-field">
      <span>{label}</span>
      {children}
    </label>
  );
}

export function FilterBar({ onClear, clearLabel = "Clear filters", children }) {
  return (
    <div className="filter-bar">
      {children}
      <button className="secondary-button" type="button" onClick={onClear}>
        {clearLabel}
      </button>
    </div>
  );
}

export function SectionSubnav({ items, activeId, onChange, label = "Section views" }) {
  return (
    <div className="section-subnav" role="tablist" aria-label={label}>
      {items.map((item) => (
        <button
          key={item.id}
          type="button"
          role="tab"
          aria-selected={activeId === item.id}
          className={activeId === item.id ? "section-subnav-button active" : "section-subnav-button"}
          onClick={() => onChange(item.id)}
        >
          <span className="section-subnav-top">
            <span className="section-subnav-label">{item.label}</span>
            {item.count != null ? <span className="section-subnav-count">{item.count}</span> : null}
          </span>
          {item.note ? <span className="section-subnav-note">{item.note}</span> : null}
        </button>
      ))}
    </div>
  );
}

export function CapabilityCard({ label, value, note }) {
  return (
    <article className="capability-card">
      <span>{label}</span>
      <strong>{value}</strong>
      <p>{note}</p>
    </article>
  );
}
