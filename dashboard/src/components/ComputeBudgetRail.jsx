import { memo } from "react";

export default memo(function ComputeBudgetRail({
  computeProfile,
  quickForms,
  updateQuickForm,
  handleSubmit,
  handleToggleContinuous,
  actionState,
  effectiveCpuBudget,
  effectiveGpuBudget,
  showMacGpuBudgetHint
}) {
  const isRunning = computeProfile.continuous_enabled !== false;
  const pendingToggle = actionState.pendingKey === "toggle continuous";
  const pendingProfile = actionState.pendingKey === "compute profile";

  return (
    <aside id="collatz-compute-rail" className="workspace-rail workspace-rail-left">
      <div className="compute-control-panel compute-control-panel-docked">
        <div className="compute-control-header">
          <span className="sidebar-kicker">Compute budget</span>
          <strong>{computeProfile.system_percent}% system</strong>
        </div>

        <div className="compute-startstop">
          <button
            className={isRunning ? "danger-button" : "primary-button"}
            onClick={handleToggleContinuous}
            disabled={pendingToggle}
            title={isRunning ? "Pause continuous compute - workers finish current run then stop queuing new ones" : "Resume continuous compute - workers will begin queuing runs again"}
          >
            {pendingToggle ? "Updating..." : isRunning ? "⏹ Stop compute" : "▶ Start compute"}
          </button>
          <p className="sidebar-note">
            {isRunning
              ? "Continuous compute is running. Workers queue runs automatically (driven by the API host)."
              : "Compute is paused: no new autopilot verification streams are queued. Finish or cancel any in-flight run, then press Start — otherwise the worker sits idle after the queue drains."}
          </p>
        </div>

        <form className="compute-sliders-form" onSubmit={handleSubmit}>
          <label className="compute-slider-row">
            <span>Whole system</span>
            <strong>{quickForms.computeSystemPercent}%</strong>
            <input
              type="range"
              min="0"
              max="100"
              step="5"
              value={quickForms.computeSystemPercent}
              onChange={(event) => updateQuickForm("computeSystemPercent", event.target.value)}
            />
          </label>
          <label className="compute-slider-row">
            <span>CPU lane</span>
            <strong>{quickForms.computeCpuPercent}%</strong>
            <input
              type="range"
              min="0"
              max="100"
              step="5"
              value={quickForms.computeCpuPercent}
              onChange={(event) => updateQuickForm("computeCpuPercent", event.target.value)}
            />
          </label>
          <label className="compute-slider-row">
            <span>GPU lane</span>
            <strong>{quickForms.computeGpuPercent}%</strong>
            <input
              type="range"
              min="0"
              max="100"
              step="5"
              value={quickForms.computeGpuPercent}
              onChange={(event) => updateQuickForm("computeGpuPercent", event.target.value)}
            />
          </label>
          {showMacGpuBudgetHint ? (
            <p className="sidebar-note compute-macos-gpu-hint" role="note">
              <strong>macOS:</strong> GPU runs (MPS/Metal) share unified memory with the CPU and the desktop compositor.
              A non-zero GPU lane can make the whole Mac feel slow. New installs default the GPU lane to 0% — raise it
              when you want <code>gpu-sieve</code> / GPU workers; keep CPU-only for a responsive UI.
            </p>
          ) : null}
          <p className="sidebar-note compute-budget-note">
            Effective now: CPU {effectiveCpuBudget}% · GPU {effectiveGpuBudget}%.
          </p>
          <button className="secondary-button" type="submit" disabled={pendingProfile}>
            {pendingProfile ? "Saving..." : "Apply budget"}
          </button>
        </form>
      </div>
    </aside>
  );
})
