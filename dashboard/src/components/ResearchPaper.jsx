import { useState, useEffect, useRef } from "react";
import { endpoints } from "../config.js";
import katex from "katex";
import "katex/dist/katex.min.css";

// Inline math detection: finds math-looking tokens in plain text and renders via KaTeX
const INLINE_MATH_RE = new RegExp(
  "(" + [
    "\\(n,\\s*T\\(n\\),\\s*T[²]\\(n\\),\\s*…\\)",  // (n, T(n), T²(n), …)
    "ν₂\\(T[ᵏ]\\([^)]+\\)\\)",                       // ν₂(Tᵏ(n))
    "T\\(n\\)\\s*=\\s*\\(3n\\+1\\)/2",                // T(n) = (3n+1)/2
    "T\\(n\\)\\s*=\\s*n/2",                            // T(n) = n/2
    "T[²ᵏ]+\\([^)]+\\)",                              // T²(n), Tᵏ(n₀)
    "T\\([^)]+\\)",                                     // T(n)
    "n\\s*∈\\s*ℕ[⁺]?",                                // n ∈ ℕ⁺
    "n\\s*≤\\s*N",                                      // n ≤ N
    "m\\s*<\\s*n",                                      // m < n
  ].join("|") + ")",
  "g"
);

function mathToLatex(match) {
  let s = match
    .replace(/²/g, "^2")
    .replace(/ᵏ/g, "^k")
    .replace(/₀/g, "_0")
    .replace(/₂/g, "_2")
    .replace(/≤/g, "\\leq ")
    .replace(/∈/g, "\\in ")
    .replace(/ℕ⁺/g, "\\mathbb{N}^+")
    .replace(/ℕ/g, "\\mathbb{N}")
    .replace(/…/g, "\\ldots")
    .replace(/ν/g, "\\nu");
  // Convert inline fractions to proper \frac notation
  s = s.replace(/\(3n\+1\)\/2/g, "\\tfrac{3n+1}{2}");
  s = s.replace(/n\/2/g, "\\tfrac{n}{2}");
  return s;
}

function renderInlineMath(text) {
  const parts = [];
  let lastIndex = 0;
  let match;
  INLINE_MATH_RE.lastIndex = 0;
  while ((match = INLINE_MATH_RE.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    const latex = mathToLatex(match[0]);
    try {
      const html = katex.renderToString(latex, { displayMode: false, throwOnError: false, trust: false });
      parts.push(<span key={match.index} className="paper-inline-katex" dangerouslySetInnerHTML={{ __html: html }} />);
    } catch (_) {
      parts.push(match[0]);
    }
    lastIndex = INLINE_MATH_RE.lastIndex;
  }
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }
  return parts;
}

// Formulas in paper.json use LaTeX syntax when prefixed with "$"
// e.g. "$\\text{HALT}(n_0) \\iff \\exists k \\in \\mathbb{N} : T^k(n_0) < n_0$"
// Plain strings (no leading $) are rendered as monospace code blocks.
function PaperFormula({ formula }) {
  const ref = useRef(null);
  const isLatex = formula.expression.startsWith("$") && formula.expression.endsWith("$");
  const latexSrc = isLatex ? formula.expression.slice(1, -1) : null;

  useEffect(() => {
    if (ref.current && isLatex) {
      try {
        katex.render(latexSrc, ref.current, {
          displayMode: true,
          throwOnError: false,
          trust: false,
        });
      } catch (_) {}
    }
  }, [latexSrc, isLatex]);

  return (
    <div className="paper-formula-block">
      {isLatex ? (
        <div className="paper-formula-katex" ref={ref} />
      ) : (
        <div className="paper-formula-expr">{formula.expression}</div>
      )}
      {formula.label && (
        <div className="paper-formula-label">({formula.label})</div>
      )}
    </div>
  );
}

function PaperDefinitionFormula({ text }) {
  const ref = useRef(null);
  // Convert the T(n) piecewise definition into LaTeX cases
  useEffect(() => {
    if (!ref.current) return;
    try {
      const lines = text.trim().split("\n").filter(Boolean);
      const cases = lines.map(l => {
        const m = l.match(/^T\(n\)\s*=\s*(.+?),\s*if\s+(.+)$/);
        if (m) {
          const cond = m[2].replace(/≡/g, "\\equiv").replace(/ℕ/g, "\\mathbb{N}").replace(/\(mod\s*(\d+)\)/g, "\\pmod{$1}").trim();
          return `${m[1].trim()} & \\text{if } ${cond}`;
        }
        return l;
      });
      const latex = `T(n) = \\begin{cases} ${cases.join(" \\\\\\\\ ")} \\end{cases}`;
      katex.render(latex, ref.current, { displayMode: true, throwOnError: false, trust: false });
    } catch (_) {
      ref.current.textContent = text;
    }
  }, [text]);
  return <div className="paper-formula-block"><div className="paper-formula-katex" ref={ref} /></div>;
}

function PaperSection({ section }) {
  const paragraphs = (section.body || "").split("\n\n").filter(Boolean);
  return (
    <section className="paper-section" id={`sec-${section.id}`}>
      <h2 className="paper-section-heading">
        <span className="paper-section-number">{section.number}.</span>
        {" "}{section.title}
      </h2>
      {paragraphs.map((para, i) => {
        if (para.startsWith("T(n)")) {
          return <PaperDefinitionFormula key={i} text={para} />;
        }
        if (para.startsWith("HALT")) {
          return <pre key={i} className="paper-code-block">{para}</pre>;
        }
        const lines = para.split("\n");
        // Ordered numbered list: "1. ... \n 2. ... \n 3. ..."
        if (lines.length > 1 && lines.every(l => /^\d+\.\s/.test(l))) {
          return (
            <ol key={i} className="paper-numbered-list" start={parseInt(lines[0], 10)}>
              {lines.map((l, j) => (
                <li key={j}>{renderInlineMath(l.replace(/^\d+\.\s*/, ""))}</li>
              ))}
            </ol>
          );
        }
        if (lines.length > 1 && lines.every(l => l.startsWith("—") || l.startsWith("-"))) {
          return (
            <ul key={i} className="paper-bullet-list">
              {lines.map((l, j) => (
                <li key={j}>{renderInlineMath(l.replace(/^[—-]\s*/, ""))}</li>
              ))}
            </ul>
          );
        }
        return <p key={i} className="paper-body">{renderInlineMath(para)}</p>;
      })}
      {section.formulas && section.formulas.map(f => (
        <PaperFormula key={f.id} formula={f} />
      ))}
      {section.entries && section.entries.length > 0 && (
        <div className="paper-entries">
          {section.entries.map((entry, i) => (
            <div key={i} className="paper-entry">
              <div className="paper-entry-header">
                <span className="paper-entry-label">{entry.label}</span>
                <span className="paper-entry-date">{entry.date}</span>
                {entry.run_id && (
                  <span className="paper-entry-run">Run #{entry.run_id}</span>
                )}
              </div>
              <p className="paper-body">{renderInlineMath(entry.body)}</p>
              {entry.formula && (
                <PaperFormula formula={{ expression: entry.formula }} />
              )}
              {entry.reproducibility && (
                <div className="paper-entry-repro">
                  <span className="paper-repro-label">Reproducibility:</span> {entry.reproducibility}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

export default function ResearchPaper() {
  const [paper, setPaper] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(endpoints.paper)
      .then(r => r.ok ? r.json() : Promise.reject(r.status))
      .then(setPaper)
      .catch(() => setError("Could not load paper.json from backend."));
  }, []);

  function handlePrint() {
    window.print();
  }

  if (error) {
    return (
      <div className="paper-page">
        <div className="paper-error">{error}</div>
      </div>
    );
  }

  if (!paper) {
    return (
      <div className="paper-page">
        <div className="paper-loading">Loading paper…</div>
      </div>
    );
  }

  return (
    <div className="paper-page">
      {/* Toolbar — hidden when printing */}
      <div className="paper-toolbar no-print">
        <div className="paper-toolbar-meta">
          <span className="paper-toolbar-version">v{paper.version}</span>
          <span className="paper-toolbar-status">{paper.status}</span>
          <span className="paper-toolbar-updated">Updated {paper.updated}</span>
        </div>
        <button
          type="button"
          className="paper-print-btn"
          onClick={handlePrint}
          title="Print / Save as PDF"
        >
          ⬇ Download PDF
        </button>
      </div>

      {/* A4 sheet */}
      <div className="paper-sheet">
        {/* Header */}
        <header className="paper-header">
          <div className="paper-classification">{paper.classification}</div>
          <h1 className="paper-title">{paper.title}</h1>
          {paper.subtitle && (
            <p className="paper-subtitle">{paper.subtitle}</p>
          )}
          <div className="paper-authors">
            {paper.authors.map((a, i) => (
              <div key={i} className="paper-author">
                <strong>{a.name}</strong>
                {a.affiliation && <span className="paper-affiliation">{a.affiliation}</span>}
              </div>
            ))}
          </div>
          <div className="paper-dateline">
            {paper.updated} · {paper.status === "living-draft" ? "Living draft — updated only on exceptional findings" : paper.status}
          </div>
        </header>

        <hr className="paper-rule" />

        {/* Abstract */}
        {paper.abstract && (
          <section className="paper-abstract-section">
            <h2 className="paper-abstract-heading">Abstract</h2>
            <p className="paper-abstract">{renderInlineMath(paper.abstract)}</p>
            {paper.keywords && (
              <p className="paper-keywords">
                <em>Keywords: </em>{paper.keywords.join(", ")}
              </p>
            )}
          </section>
        )}

        <hr className="paper-rule" />

        {/* Sections */}
        {paper.sections.map(section => (
          <PaperSection key={section.id} section={section} />
        ))}

        <hr className="paper-rule" />

        {/* References */}
        {paper.references && paper.references.length > 0 && (
          <section className="paper-section paper-references">
            <h2 className="paper-section-heading">References</h2>
            <ol className="paper-ref-list">
              {paper.references.map(ref => (
                <li key={ref.id} className="paper-ref-item">{ref.text}</li>
              ))}
            </ol>
          </section>
        )}

        {/* Footer */}
        <footer className="paper-footer">
          <div className="paper-footer-line">
            Collatz Attack Lab · {paper.updated} · v{paper.version}
          </div>
          <div className="paper-footer-line paper-footer-note">
            This is a living research document. Only findings that are independently reproducible, rigorously verified, and of exceptional significance appear in Section 5.
          </div>
        </footer>
      </div>
    </div>
  );
}
