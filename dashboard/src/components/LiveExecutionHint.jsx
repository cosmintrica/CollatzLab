import {
  classifyWorkspaceHost,
  extractHostPlatform,
  getHintCodeSnippet,
} from "../lib/workspaceCliHints.js";

/**
 * Explains why Live Math shows zero active/historical runs: the ledger and worker are separate from hardware detection.
 * Copy is driven by the API host (hardware inventory), not the browser, so Windows/macOS/Linux stay correct.
 *
 * @param {{ totalRuns: number, queuedRuns: number, runningRuns: number, hardwareInventory?: unknown }} props
 */
export default function LiveExecutionHint({ totalRuns, queuedRuns, runningRuns, hardwareInventory }) {
  const hasQueue = queuedRuns > 0;
  const isIdleQueue = hasQueue && runningRuns === 0;
  const emptyLedger = totalRuns === 0;

  if (!emptyLedger && !isIdleQueue) {
    return null;
  }

  const host = extractHostPlatform(hardwareInventory);
  const platform = classifyWorkspaceHost(host);
  const devOk = platform.devScriptsSupported;
  const code = getHintCodeSnippet(platform, { emptyLedger });

  const unsupportedBanner =
    !devOk && platform.unsupportedNote ? (
      <p className="live-execution-hint-warn" role="alert">
        <strong>Limited platform guidance.</strong> {platform.unsupportedNote}
      </p>
    ) : null;

  return (
    <div
      className={
        devOk ? "live-execution-hint" : "live-execution-hint live-execution-hint--unsupported"
      }
      role="status"
    >
      <strong>No Collatz runs in motion</strong>
      {unsupportedBanner}
      {emptyLedger ? (
        <p>
          Hardware detection only describes {platform.devicePhrase} — it does not enqueue work. Create a run from the{" "}
          <strong>Operations</strong> tab (queue), or from the terminal (example below), then start a{" "}
          <strong>worker</strong> so queued runs execute.
        </p>
      ) : (
        <p>
          You have <strong>{queuedRuns}</strong> queued run(s) but nothing is <strong>running</strong>. Start a worker
          process on {platform.devicePhrase} (it uses the project <code>.venv</code> when present).
        </p>
      )}
      <pre className="live-execution-hint-code">{code}</pre>
      <p className="live-execution-hint-note">
        API at <code>127.0.0.1:8000</code> serves data; the worker is a separate process that claims runs from SQLite.
      </p>
    </div>
  );
}
