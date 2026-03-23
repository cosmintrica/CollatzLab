import { liveMathSections } from "../config.js";

export default function LiveMathNavigator({ runs, selectedRun, onSelectRun, onJumpToSection }) {
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
