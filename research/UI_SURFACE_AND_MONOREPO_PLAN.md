# UI surfaces & monorepo plan (public site vs worker console)

This document extends [`FEDERATED_LAB.md`](./FEDERATED_LAB.md) and [`ROADMAP.md`](./ROADMAP.md) with a **technical** blueprint: one **monorepo**, two **deployable UI personalities**, and how they relate to the future **coordinator** API.

## 1. Is a monorepo “too complicated”?

**No** — for this project a monorepo is a good default.

| Concern | Monorepo answer |
|--------|------------------|
| Two different UIs | Solve with **build-time shell** (`public` vs `worker`), not with two repos. |
| Access control | **CODEOWNERS** on paths (`dashboard/`, `backend/`); optional separate **deploy workflows** (`deploy-public.yml` vs `deploy-worker.yml`). |
| Release coupling | Use **semantic versioning** for API; worker bundle pins `VITE_API_BASE_URL` to a coordinator version you control. |
| Bundle size for workers | Worker build should **exclude** heavy tabs (Paper, Community) via entry/config — see §4. |

What *is* genuinely complex (independent of mono vs multi repo):

- **Coordinator API**: auth, job leases, uploads, trust/quarantine ([`FEDERATED_LAB.md`](./FEDERATED_LAB.md)).
- **Contract stability** between coordinator and worker CLI/UI.
- **Operational security** (tokens, CORS, rate limits).

Those stay the same whether code lives in one repo or three.

---

## 2. Product intent: two views

### 2.1 Public site (maintainer-controlled narrative)

**Audience:** visitors, researchers browsing evidence, reading directions, community surfaces.

**Include (current tab ids from `dashboard/src/config.js`):**

| Tab id | Include | Notes |
|--------|---------|--------|
| `overview` | Yes | Landing / lab story |
| `live-math` | Yes | Public transparency of active compute (optional subset) |
| `directions` | Yes | Full tracks / research directions |
| `community` | Yes | Reddit, external discourse |
| `evidence` | Yes | Full claims, runs, artifacts, linking |
| `queue` | Yes (read-oriented for public) | May later hide destructive actions for anonymous users |
| `paper` | Yes | Research paper view |
| `guide` | Yes | Onboarding |

**Rails/components to keep:** Reddit intel, source review flows, LLM/autopilot surfaces (subject to your policy), full evidence drill-down.

### 2.2 Worker console (contributor / operator)

**Audience:** someone running **local or federated compute** for the lab — needs clarity, queue, hardware, logs, checkpoints — **not** the full research/community product.

**Default policy (matches your request): *same capabilities as today, minus narrative / social / paper*.**

| Tab id | Worker shell | Rationale |
|--------|--------------|-----------|
| `overview` | **Yes (compact)** | Status, link to coordinator health, worker identity — not full marketing copy |
| `live-math` | **Yes** | Trace / ledger / records for active runs |
| `directions` | **Optional / reduced** | **Phase 1:** hide or show **Verification** only; hide sandbox lemma/hypothesis tools until needed |
| `community` | **No** | No Reddit, no community tab |
| `evidence` | **Yes (subset)** | Runs, validated results, artifacts useful for debugging; hide heavy claim curation if desired |
| `queue` | **Yes** | Operations: compute profile, queue, workers — core |
| `paper` | **No** | |
| `guide` | **Yes (operator)** | Short “how to run worker / read queue” — can reuse part of Guide or a dedicated mini-doc |

**Explicitly exclude from worker shell:**

- `paper`
- `community` (and **RedditIntelRail** / Reddit API usage where it is only for that tab)
- Source-review–only flows that are not needed for compute (can stay in API but unused in UI)

**Keep:**

- **ComputeBudgetRail**, **RunRail**, **LogsPanel**, hardware/workers endpoints, compute profile — operator essentials.

---

## 3. Backend / API alignment (now vs coordinator future)

### Today (local-first)

- Single FastAPI + SQLite; worker processes use CLI + optional dashboard on `localhost`.
- Worker shell can point `VITE_API_BASE_URL` at the same origin as the maintainer’s browser — still **one API**.

### Future (federated)

| Layer | Public site | Worker console |
|-------|-------------|----------------|
| **API host** | Your **coordinator** (read-heavy + curated writes) | Same coordinator **or** a dedicated **worker-facing** subdomain |
| **Auth** | Session / OAuth for humans | **Machine token** or mTLS for `worker-agent` |
| **Endpoint sets** | Full read + maintainer actions | **Subset**: jobs poll/lease, run status, checkpoint upload, capabilities register — optional read-only summary |

**Important:** hiding tabs in the UI is **not** security. The coordinator must **enforce** scopes:

- e.g. token role `worker` cannot call `POST .../reddit/...` or expensive LLM routes.

Document per-route `required_role: public | maintainer | worker` when you add auth.

---

## 4. Implementation strategy (phased, monorepo)

### Phase W1 — Configuration shell (low risk)

1. Add **`VITE_COLLATZ_UI_SHELL`** with values `public` | `worker` (default `public` for backward compatibility).
2. In `dashboard/src/config.js` (or a new `shell.js`):
   - Export `tabsForShell(shell)` derived from a static map (§2 tables).
   - Export `featureFlags`: e.g. `enableReddit`, `enablePaper`, `enableCommunityTab`, `enableSourceReview`.
3. In `App.jsx`:
   - Read shell once from `import.meta.env.VITE_COLLATZ_UI_SHELL`.
   - Render tab bar from filtered list; **guard** heavy sections with flags so Reddit/Paper components never mount in worker shell (saves work + avoids accidental fetches).

**Deliverable:** two local builds:

```bash
# public (default)
npm run build

# worker console
VITE_COLLATZ_UI_SHELL=worker npm run build
```

Document in `dashboard/README.md` (or root README).

### Phase W2 — Smaller worker bundle (optional)

- Add **`vite.config.worker.ts`** (or second `build` script) with different `define` / entry `worker-main.jsx` that imports a slimmer `AppWorker.jsx` if tree-shaking is insufficient.
- Only needed if bundle analysis shows large dead chunks (e.g. paper JSON, community).

### Phase W3 — Deploy pipelines

- GitHub Actions: two jobs or workflow_dispatch inputs — **artifact `dist-public`** vs **`dist-worker`** with env set.
- Public: deploy to Vercel/Netlify; Worker: optional **private** static host or bundled inside a desktop wrapper later.

### Phase W4 — Coordinator + token scopes

- Implement API auth and **route allowlists** for `worker` role.
- Worker CLI already exists; extend to **register + poll** coordinator ([`FEDERATED_LAB.md`](./FEDERATED_LAB.md) milestones).

---

## 5. Testing & regression

- **E2E or smoke:** build `worker` shell, assert URL hash routes for disabled tabs 404 or redirect to `overview`.
- **Unit:** `tabsForShell('worker')` does not include `paper` / `community`.
- **API tests (later):** worker token rejected on `/api/external/reddit/...`.

---

## 6. Relation to existing docs

| Document | Role |
|----------|------|
| [`FEDERATED_LAB.md`](./FEDERATED_LAB.md) | Trust model + three roles (dashboard / coordinator / worker-agent) |
| [`ROADMAP.md`](./ROADMAP.md) Phase 5–6 | Hosting + federated compute |
| [`DEVELOPMENT_BACKLOG.md`](./DEVELOPMENT_BACKLOG.md) §7–10 | Hosting readiness + federated milestones |
| [`docs/STATUS_AND_NEXT_STEPS.md`](../docs/STATUS_AND_NEXT_STEPS.md) | Current handoff (Metal spike, etc.) |

**Suggested backlog additions (you can paste into `DEVELOPMENT_BACKLOG.md`):**

- Implement `VITE_COLLATZ_UI_SHELL` + tab/feature map (Phase W1).
- Document dual build in README / `dashboard/README.md`.
- Add smoke test for worker shell tab list.
- (Later) Coordinator route scopes for `worker` vs `public`.

---

## 7. Summary

- **Monorepo is OK** and keeps API + both UIs in sync.
- **Two views** = same codebase, **different shell** via env + filtered navigation; optionally second Vite entry later.
- **Worker UI** = current operator value **minus** Paper, Community/Reddit, and (recommended) trimmed Tracks until needed.
- **Real enforcement** of “worker cannot hit social/LLM” happens on the **coordinator**, not only in the UI.
