import { memo } from "react";

export default memo(function ComputeBudgetRail({
  computeProfile,
  quickForms,
  updateQuickForm,
  handleSubmit,
  handleToggleContinuous,
  actionState,
  effectiveCpuBudget,
  effectiveGpuBudget
}) {
  const isRunning = computeProfile.continuous_enabled !== false;
  const pendingToggle = actionState.pendingKey === "toggle continuous";
  const pendingProfile = actionState.pendingKey === "compute profile";

  return (
    <aside className="workspace-rail workspace-rail-left">
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
              ? "Continuous compute is running. Workers queue runs automatically."
              : "Compute is paused. Current run finishes, then workers go idle."}
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
