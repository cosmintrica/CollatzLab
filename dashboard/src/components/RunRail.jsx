import { StatusPill } from "./ui.jsx";
import { describeRunPurpose, describeRunStatusDetail, runRangeMagnitude, formatCompactTimestamp } from "../utils.js";
import { runProgress, runTimingMicroSuffix } from "../orbitCalc.js";

import { memo } from "react";

export default memo(function RunRail({ runs, selectedRunId, onSelectRun, speedPollRef }) {
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
            const purpose = describeRunPurpose(run);
            const statusDetail = describeRunStatusDetail(run);
            const magnitude = runRangeMagnitude(run);
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
                <span className="ts-micro run-rail-ts">
                  {formatCompactTimestamp(run.finished_at || run.started_at || run.created_at)}
                  {runTimingMicroSuffix(run, { pollRef: speedPollRef })}
                </span>
                <span className="run-rail-name">{run.name}</span>
                <span className="run-rail-meta">{run.kernel} | {run.hardware}</span>
                <span className="run-rail-purpose">{purpose}</span>
                {statusDetail ? (
                  <span className={`run-rail-detail ${run.status === "failed" ? "failure" : ""}`}>{statusDetail}</span>
                ) : null}
                {magnitude ? <span className="run-rail-range power">{magnitude}</span> : null}
                <span className="run-rail-range">
                  {run.range_start.toLocaleString()} → {run.range_end.toLocaleString()}
                </span>
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
})
