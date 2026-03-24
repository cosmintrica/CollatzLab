# Status handoff: Metal spike + GitHub / hosting topology

Operational notes so we know **where we paused** and **what to do next**, without relying on chat history.

**Verified startup:** before a live session, follow [`PLATFORM_VERIFY_AND_START.md`](./PLATFORM_VERIFY_AND_START.md) (`bash scripts/verify-platform.sh` or `--full`).

---

## 1. Metal toolchain & native performance spike

### Where we are

- **`scripts/metal_native_spike/`** — exploratory **Swift + Metal** benchmark (not production lab code). Measures rough **odd seeds / second** for a uint64 odd-descent kernel.
- **`run.sh`** — compiles `CollatzSpike.metal` → `.metallib` and builds the Swift driver (requires **Metal Toolchain**).
- **`compare_with_lab.py`** — runs the spike and **`gpu-sieve` (PyTorch MPS)** on the **same odd workload** and prints a ratio.

### Metal Toolchain

If `xcrun metal` fails with *missing Metal Toolchain*:

```bash
xcodebuild -downloadComponent MetalToolchain
xcrun -sdk macosx metal --version
```

**Status:** toolchain installed on maintainer Mac — `run.sh` builds and runs successfully.

**Sample spike (Apple M1 Pro, 500k odds, exploratory kernel):** JSON reported on the order of **~1.3–1.6×10⁸ odd/s** and **wall_s ~3–4 ms** for the timed pass after warmup. Numbers **jitter** when `wall_s` is tiny; use a larger `--count` (e.g. 5M) for stabler throughput estimates.

### Checklist

1. ~~`bash scripts/metal_native_spike/run.sh --count 500000 --base 1`~~ — OK when toolchain present.
2. **Full Metal vs MPS ratio:** from repo root, `python3 scripts/metal_native_spike/compare_with_lab.py` **auto re-execs** into `.venv/bin/python` when present (or pass `--python`). Optional: `--preset-env tuned|extreme`, larger `--count` for stabler numbers.

**Sample (Apple M1 Pro, 2M odds, `tuned` MPS):** Metal spike ~**8.6×10⁷ odd/s** vs lab `gpu-sieve` ~**1.9×10⁴ odd/s** — ratio ~**10³–10⁴×** is expected: spike = bare kernel; lab path = PyTorch + batches + sync + Python aggregates (see `scripts/metal_native_spike/README.md`).
3. Record numbers if useful; **decision**: whether a full native Metal port is worth it vs staying on **MPS + Python** (spike kernel ≠ byte-identical lab `gpu-sieve`).

**Stuck `gpu-sieve` run:** stop worker → `bash scripts/recover-stuck-run.sh COL-XXXX` or `--migrate-cpu-sieve` — see [`docs/RUN_RECOVERY.md`](./RUN_RECOVERY.md) (**Metal = GPU only**; for escape hatch use **`cpu-sieve` on CPU**, not “CPU on Metal”).

### User-facing installs

End users of the **normal lab** (Python + dashboard) **do not** need the Metal Toolchain. It is only for **compiling** `.metal` during dev/CI. Runtime uses the OS GPU stack; a shipped native app would bundle a **prebuilt** `.metallib`.

---

## 2. GitHub layout: centralized UI vs workers (existing plan in-repo)

The architecture you described is already captured here:

| Document | What it covers |
|----------|----------------|
| [`research/FEDERATED_LAB.md`](../research/FEDERATED_LAB.md) | **Public dashboard** (e.g. Vercel) ↔ **maintainer-owned coordinator + DB** ↔ **open-source worker-agent** on user machines; trust model. |
| [`research/ROADMAP.md`](../research/ROADMAP.md) § Phase 5–6 | **Remote hosting**: Vite + `VITE_API_BASE_URL`; dashboard separate from API; **federated compute** milestones. |
| [`research/DEVELOPMENT_BACKLOG.md`](../research/DEVELOPMENT_BACKLOG.md) § 9–10 | Backlog: coordinator, remote workers, hosted main platform; **three roles** (dashboard / coordinator / workers). |
| [`research/WORKER_QUEUE.md`](../research/WORKER_QUEUE.md) | Queue / execution contract (local today; aligns with future remote leases). |

Dashboard already supports a remote API via **`VITE_API_BASE_URL`** (see `dashboard/src/config.js`).

**Preferred direction:** stay on a **monorepo** and add two **UI shells** (`public` vs `worker`) — see [`research/UI_SURFACE_AND_MONOREPO_PLAN.md`](../research/UI_SURFACE_AND_MONOREPO_PLAN.md) for the full technical matrix (tabs, env var, phases, API scopes).

### Repo shape options (when you split for GitHub)

**A — Monorepo (current)**  
- Single repo: `backend/`, `dashboard/`, `research/`, `scripts/`.  
- **Pros:** one PR crosses API + UI; shared CI; matches today.  
- **Cons:** permissions are all-or-nothing unless you use [GitHub CODEOWNERS](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners) and path filters.

**B — Multi-repo under one GitHub org** (fits “I control the central frontend / API”)  
Example split (names illustrative):

| Repo | Owner / deploy | Notes |
|------|----------------|--------|
| `collatz-lab-dashboard` | You — static host (Vercel, etc.) | Only static build; `VITE_API_BASE_URL` → your coordinator. |
| `collatz-lab-coordinator` | You — private or public API + DB | FastAPI, migrations, auth; **canonical** data. |
| `collatz-lab` or `collatz-lab-worker` | Public OSS | CLI + worker loop; users run locally; talks to coordinator when federated mode exists. |

**Pros:** clear access control (you push to dashboard/coordinator repos; contributors fork worker). **Cons:** coordinated releases (API version vs UI expectations), duplicate CI boilerplate unless templated.

**C — Monorepo + GitHub Actions matrix**  
- Stay in one repo; use workflows with `paths:` filters so `dashboard/**` only triggers frontend CI, `backend/**` backend CI.  
- **CODEOWNERS** so your account owns `dashboard/` and coordinator-related paths.

### Practical GitHub features (industry pattern)

- **Organization** (`github.com/collatz-lab` or similar): separate repos or teams (Maintainers vs Contributors).
- **Environments + secrets** for deploy (Vercel token, API host, DB URLs) — never commit secrets; `.env` stays gitignored ([README](../README.md) already notes this).
- **Branch protection** on `main` for repos you control; worker repo can stay permissive for OSS contributions.

### Not done yet (product-wise)

- Remote **coordinator** API (job lease, upload, auth) — still **local-first** SQLite + same-process workers.
- Packaged **`worker-agent`** that only polls your hosted API — outlined in FEDERATED_LAB / ROADMAP, not implemented as a separate deployable artifact.

---

## 3. Quick “resume from here” list

1. Finish **Metal Toolchain** install; run spike + `compare_with_lab.py`.
2. Decide **monorepo vs split repos** using the table above; if split, add a short **`research/HOSTING_TOPOLOGY.md`** with final repo names and deploy targets.
3. When implementing federation, treat **`FEDERATED_LAB.md`** milestones as the checklist (registration → poll → shard → upload → trust).
