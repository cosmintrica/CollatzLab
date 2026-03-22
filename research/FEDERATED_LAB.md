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
