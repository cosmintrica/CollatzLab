export const apiBase = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export const endpoints = {
  summary: `${apiBase}/api/summary`,
  directions: `${apiBase}/api/directions`,
  tasks: `${apiBase}/api/tasks`,
  runs: `${apiBase}/api/runs`,
  claims: `${apiBase}/api/claims`,
  hypotheses: `${apiBase}/api/hypotheses`,
  hypothesisBattery: `${apiBase}/api/hypotheses/battery`,
  hypothesisBatteryStability: `${apiBase}/api/hypotheses/battery/stability`,
  claimRunLinks: `${apiBase}/api/claim-run-links`,
  linkClaimRun: `${apiBase}/api/claims/link-run`,
  sources: `${apiBase}/api/sources`,
  artifacts: `${apiBase}/api/artifacts`,
  artifactContent: (artifactId) => `${apiBase}/api/artifacts/${artifactId}/content`,
  artifactDownload: (artifactId) => `${apiBase}/api/artifacts/${artifactId}/download`,
  consensusBaseline: `${apiBase}/api/consensus-baseline`,
  computeProfile: `${apiBase}/api/compute/profile`,
  llmStatus: `${apiBase}/api/llm/status`,
  llmSetup: `${apiBase}/api/llm/setup`,
  llmAutopilotStatus: `${apiBase}/api/llm/autopilot/status`,
  llmAutopilotConfig: `${apiBase}/api/llm/autopilot/config`,
  llmAutopilot: `${apiBase}/api/llm/autopilot/run`,
  redditFeed: `${apiBase}/api/external/reddit/collatz?limit=10`,
  fallacyTags: `${apiBase}/api/review/fallacy-tags`,
  sourceReviews: (sourceId) => `${apiBase}/api/sources/${sourceId}/reviews`,
  sourceReviewDraft: (sourceId) => `${apiBase}/api/sources/${sourceId}/review-draft`,
  modularProbe: `${apiBase}/api/review/probes/modular`,
  hardware: `${apiBase}/api/workers/capabilities`,
  workers: `${apiBase}/api/workers`,
  gpuSieveMetalStdioShutdown: `${apiBase}/api/workers/gpu-sieve-metal/stdio-shutdown`,
  sourceDelete: (sourceId) => `${apiBase}/api/sources/${sourceId}`,
  paper: `${apiBase}/api/paper`,
  logs: `${apiBase}/api/logs`,
  benchMetalChunkPresets: `${apiBase}/api/bench/metal-chunk/presets`,
  benchMetalChunkStatus: `${apiBase}/api/bench/metal-chunk/status`,
  benchMetalChunkHistory: (limit = 25) => `${apiBase}/api/bench/metal-chunk/history?limit=${limit}`,
  benchMetalChunkRun: `${apiBase}/api/bench/metal-chunk/run`,
  benchMetalChunkRunDetail: (jobId) => `${apiBase}/api/bench/metal-chunk/runs/${jobId}`,
  benchMetalChunkHallOfFame: (platform, limit = 30) =>
    `${apiBase}/api/bench/metal-chunk/hall-of-fame?platform=${encodeURIComponent(platform)}&limit=${limit}`
};

export const tabs = [
  { id: "overview", label: "Start Here" },
  { id: "live-math", label: "Live Math" },
  { id: "directions", label: "Tracks" },
  { id: "community", label: "Community" },
  { id: "evidence", label: "Evidence" },
  { id: "queue", label: "Operations" },
  { id: "benchmark", label: "Benchmark" },
  { id: "paper", label: "Paper" },
  { id: "guide", label: "Guide" }
];

export const liveMathSections = [
  { id: "live-trace", label: "Trace" },
  { id: "live-ledger", label: "Ledger" },
  { id: "live-records", label: "Records" }
];

export const defaultDirectionOptions = [
  { slug: "verification", title: "Verification" },
  { slug: "inverse-tree-parity", title: "Inverse Tree Parity" },
  { slug: "lemma-workspace", title: "Lemma Workspace" },
  { slug: "two-adic-v2", title: "2-adic / v2 Explorer" },
  { slug: "hypothesis-sandbox", title: "Hypothesis Sandbox" }
];

export const sourceTypeOptions = [
  "peer_reviewed",
  "preprint",
  "self_published",
  "forum",
  "blog",
  "qa",
  "wiki",
  "media",
  "internal"
];

export const sourceClaimTypeOptions = [
  "open_problem_consensus",
  "partial_result",
  "computational_verification",
  "proof_attempt",
  "heuristic",
  "discussion"
];

export const sourceStatusOptions = ["intake", "under_review", "flagged", "supported", "refuted", "context"];

export const mapVariantOptions = ["unspecified", "standard", "shortcut", "odd_only", "inverse_tree"];

export const llmModelSuggestions = [
  "gemini-2.5-flash",
  "gemini-2.5-flash-lite",
  "gemini-flash-latest",
  "gemini-2.5-pro",
  "gemini-3-flash-preview"
];

export const rubricFieldOptions = [
  { key: "peer_reviewed", label: "Peer reviewed" },
  { key: "acknowledged_errors", label: "Acknowledged errors" },
  { key: "defines_map_variant", label: "Defines map variant" },
  { key: "distinguishes_empirical_from_proof", label: "Separates proof from evidence" },
  { key: "proves_descent", label: "Proves descent" },
  { key: "proves_cycle_exclusion", label: "Proves cycle exclusion" },
  { key: "uses_statistical_argument", label: "Uses statistical argument" },
  { key: "validation_backed", label: "Validation backed" }
];

export const defaultFallacyCatalog = [
  {
    tag: "empirical-not-proof",
    label: "Empirical is not proof",
    description: "Finite computation is evidence, not a universal theorem."
  },
  {
    tag: "almost-all-not-all",
    label: "Almost all is not all",
    description: "Density results do not settle every integer."
  },
  {
    tag: "circular-descent",
    label: "Circular descent",
    description: "The source assumes the same global descent it claims to prove."
  },
  {
    tag: "unchecked-generalization",
    label: "Unchecked generalization",
    description: "A local pattern is promoted to all n without a valid universal step."
  },
  {
    tag: "reverse-tree-gap",
    label: "Reverse tree gap",
    description: "The inverse-tree picture is not enough without a forward implication."
  },
  {
    tag: "publishing-does-not-imply-validity",
    label: "Publication is not validation",
    description: "Posting or publishing a manuscript does not make it correct."
  },
  {
    tag: "variant-confusion",
    label: "Variant confusion",
    description: "Standard, shortcut, odd-only, or inverse-tree variants are mixed together."
  },
  {
    tag: "proof-by-large-search",
    label: "Proof by large search",
    description: "A large verified interval is treated as if it solved the problem."
  },
  {
    tag: "statistical-leap",
    label: "Statistical leap",
    description: "Average-case language is used to conclude a deterministic theorem."
  }
];

export const claimRunRelationOptions = ["supports", "tests", "refutes", "motivates", "depends_on"];

export const directionGuide = {
  verification: {
    label: "Evidence track",
    role:
      "Runs CPU/GPU sweeps (Numba, native cpu-sieve where built, MPS/Metal gpu-sieve on macOS when configured), compares kernels, and falsifies weak heuristics.",
    success: "Finds reproducible evidence or real search-space reduction.",
    caution: "This is not the proof track by itself.",
    evidence: "run-heavy lane"
  },
  "inverse-tree-parity": {
    label: "Structure track",
    role: "Explores odd predecessors, parity vectors, and modular filters.",
    success: "Finds structural constraints that survive wider testing.",
    caution: "Reverse-tree intuition still needs a forward implication.",
    evidence: "artifact-heavy lane"
  },
  "lemma-workspace": {
    label: "Proof track",
    role: "Tracks lemmas, dependencies, counterexamples, and source review.",
    success: "Promotes exact claims toward formalization with linked evidence.",
    caution: "Claims stay provisional until evidence and review agree.",
    evidence: "claim-heavy lane"
  },
  "two-adic-v2": {
    label: "2-adic track",
    role: "Studies v2(3n+1), odd-only compression, and 2-adic residue structure.",
    success: "Finds stable odd-step or valuation regularities that survive extension.",
    caution: "2-adic patterns are signals, not proofs, until they produce a falsifiable next step.",
    evidence: "odd-step structural lane"
  },
  "hypothesis-sandbox": {
    label: "Hypothesis track",
    role:
      "Worker-rotated hypothesis probes, optional full battery and battery stability report from Operations → Theory, queued sandbox runs (including kernel probes and experimental kernels such as cpu-barina).",
    success: "Produces falsifiable hypotheses, plausible anomalies, or cleanly refuted ideas that sharpen the search.",
    caution:
      "Outputs here are experimental ideas and evidence artifacts, not supported claims by default; Barina metrics are not the same contract as standard sieve validation.",
    evidence: "hypothesis and probe lane"
  }
};

export const evidenceGuide = [
  {
    kind: "validated-result",
    title: "Validated result",
    detail:
      "A run that passed an independent replay against the agreed reference (for sieve-family kernels, odd-descent vs SoT). Strong computational evidence — still not a proof of the global conjecture."
  },
  {
    kind: "claim",
    title: "Claim",
    detail: "A mathematical statement. It is theory, not evidence, until runs or artifacts support it."
  },
  {
    kind: "artifact",
    title: "Artifact",
    detail: "A saved file: JSON output, report, note, or proof draft you can preview or download."
  },
  {
    kind: "run",
    title: "Raw run",
    detail: "A compute record. Useful, but lower-trust than a validated result until replay succeeds."
  }
];
