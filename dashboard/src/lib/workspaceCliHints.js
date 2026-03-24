/**
 * Dev-workflow copy for Live Math hints: tied to the API host (hardware inventory),
 * not the browser OS — keeps Windows/macOS/Linux guidance accurate and modular.
 */

/** @typedef {{ os?: string, is_windows?: boolean, is_linux?: boolean, is_darwin?: boolean, is_arm64?: boolean, machine?: string }} HostPlatform */

/**
 * @param {unknown} hardwareInventory
 * @returns {HostPlatform | null}
 */
export function extractHostPlatform(hardwareInventory) {
  if (!Array.isArray(hardwareInventory)) {
    return null;
  }
  for (const item of hardwareInventory) {
    const host = item?.metadata?.host;
    if (host && typeof host === "object") {
      return host;
    }
  }
  return null;
}

/**
 * @param {HostPlatform | null} host
 * @returns {{
 *   tier: 'windows' | 'darwin' | 'linux' | 'unknown' | 'other',
 *   devicePhrase: string,
 *   shell: 'powershell' | 'bash' | null,
 *   devScriptsSupported: boolean,
 *   unsupportedNote: string | null,
 * }}
 */
export function classifyWorkspaceHost(host) {
  if (!host || typeof host !== "object") {
    return {
      tier: "unknown",
      devicePhrase: "the computer running the API",
      shell: null,
      devScriptsSupported: false,
      unsupportedNote:
        "Hardware inventory is not available yet (refresh or check the API). When it loads, hints will match the host OS.",
    };
  }

  if (host.is_windows === true) {
    return {
      tier: "windows",
      devicePhrase: "this Windows PC",
      shell: "powershell",
      devScriptsSupported: true,
      unsupportedNote: null,
    };
  }

  if (host.is_darwin === true) {
    return {
      tier: "darwin",
      devicePhrase: "this Mac",
      shell: "bash",
      devScriptsSupported: true,
      unsupportedNote: null,
    };
  }

  if (host.is_linux === true) {
    return {
      tier: "linux",
      devicePhrase: "this Linux machine",
      shell: "bash",
      devScriptsSupported: true,
      unsupportedNote: null,
    };
  }

  const os = String(host.os || "unknown").toLowerCase();
  return {
    tier: "other",
    devicePhrase: "this system",
    shell: null,
    devScriptsSupported: false,
    unsupportedNote: `Collatz Lab shell scripts are maintained for Windows, macOS, and Linux. This API host reports OS “${os}”. You can still queue runs from Operations; use Python’s \`collatz_lab.cli\` if you installed the backend elsewhere — see the repository README.`,
  };
}

const BASH_DEMO = `# One-shot: enqueue a tiny CPU run, then execute it once
bash scripts/demo_enqueue_and_worker_once.sh`;

const BASH_WORKER = `bash scripts/run-worker.sh`;

/** Portable fallback when shell scripts are not documented for the host. */
const PYTHON_DEMO = `# One-shot (any OS with the backend installed): init, enqueue, run once
python -m collatz_lab.cli init
python -m collatz_lab.cli run start --direction verification --name demo-local-smoke --start 1 --end 2000 --kernel cpu-parallel --hardware cpu --enqueue-only
python -m collatz_lab.cli worker once --name demo-worker --hardware auto`;

const WINDOWS_PS_DEMO = `# One-shot (PowerShell, repo root). Prefer .venv when present:
# .\\.venv\\Scripts\\python.exe -m collatz_lab.cli ...
python -m collatz_lab.cli init
python -m collatz_lab.cli run start --direction verification --name demo-local-smoke --start 1 --end 2000 --kernel cpu-parallel --hardware cpu --enqueue-only
python -m collatz_lab.cli worker once --name demo-worker --hardware auto`;

const WINDOWS_PS_WORKER = `powershell -ExecutionPolicy Bypass -File .\\scripts\\run-worker.ps1`;

/**
 * @param {ReturnType<typeof classifyWorkspaceHost>} classification
 * @param {{ emptyLedger: boolean }} ctx
 * @returns {string}
 */
export function getHintCodeSnippet(classification, ctx) {
  const { tier, shell, devScriptsSupported } = classification;

  if (!ctx.emptyLedger) {
    if (tier === "windows" && devScriptsSupported) {
      return WINDOWS_PS_WORKER;
    }
    if (devScriptsSupported && shell === "bash") {
      return BASH_WORKER;
    }
    return `python -m collatz_lab.cli worker start --name local-worker --hardware auto --poll-interval 5`;
  }

  if (tier === "windows" && devScriptsSupported) {
    return WINDOWS_PS_DEMO;
  }
  if (devScriptsSupported && shell === "bash") {
    return BASH_DEMO;
  }
  return PYTHON_DEMO;
}
