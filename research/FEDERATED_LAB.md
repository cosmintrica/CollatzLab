# Federated Lab Model

## Intended future shape

The project remains open source, but the operational topology is centralized:

- one main platform is controlled by the maintainer;
- the maintainer-controlled platform is the canonical source of truth;
- outside contributors can run a separate `worker-agent` on their own computers;
- worker results are uploaded back to the maintainer-controlled coordinator and database.

## Role split

### 1. Public dashboard

- can be hosted separately, for example on Vercel;
- shows the current state of the lab;
- does not perform heavy compute;
- talks to the coordinator API.

### 2. Coordinator

- owned and operated by the maintainer;
- stores the canonical database;
- assigns bounded jobs;
- receives uploaded results and artifacts;
- tracks provenance, trust, and deduplication;
- decides what must be revalidated independently.

### 3. Worker-agent

- open source and installable by anyone;
- runs on the contributor's personal machine;
- detects local hardware capabilities;
- requests bounded jobs from the coordinator;
- executes local compute;
- uploads outputs, metadata, and checkpoints back to the coordinator.

## Open source vs “who hosts the public site”

**Open source** means the **license and availability of source code** others can inspect, rebuild, and fork (e.g. Apache-2.0). It does **not** require that your **production deployment**, secrets, or CI logs be public.

- **Public monorepo** (backend + dashboard + worker in one public repo): strongest story for *“the UI I see is the same code as on GitHub”* — no suspicion of a hidden fork of the dashboard.
- **Private hosting repo** (only Terraform, DNS, Vercel project config): compatible with an open source **application** repo; you are not hiding the app logic if the app code stays public elsewhere.
- **Private application code** (dashboard closed): the project can still be “partially open source,” but visitors cannot verify that the live site matches any public artifact — easier to raise *appearance* concerns even if tampering would really happen on the **API/DB** side.

**Record integrity** (accusations of manipulating results) is only weakly addressed by open-sourcing the frontend. What matters more:

- coordinator + validation paths + DB rules are **auditable**;
- important results are **reproducible** (parameters published, independent replay);
- optional **exports / checkpoints / signed bundles** so third parties can mirror or diff history.

So: **yes, you can centralize hosting and keep the project open source**; for **trust**, prefer **public source for all code that interprets or presents canonical data**, plus transparent **data policy** (exports, replay, quarantine rules already in this doc).

## Trust model

Remote compute is useful, but it is not automatically trusted.

Rules:

- every remote result must carry capability and environment metadata;
- large or important results should be replayed on independent hardware;
- a remote result should not directly promote a claim without revalidation;
- duplicate or contradictory submissions should be quarantined until reviewed.

## Why this model

This gives the project both:

- openness: anyone can inspect the code and contribute compute;
- control: one canonical platform centralizes results and keeps the research state coherent.

## Future implementation milestones

1. Worker registration and capability handshake
2. Job polling and lease model
3. Shard assignment for compute runs
4. Artifact/result upload
5. Trust scoring and replay rules
6. Multi-machine validation before promotion

## See also

- [`UI_SURFACE_AND_MONOREPO_PLAN.md`](./UI_SURFACE_AND_MONOREPO_PLAN.md) — how the **public dashboard** vs **worker console** share one repo (env-driven UI shell, phased builds, coordinator token scopes).
