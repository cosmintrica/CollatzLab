# Local workspace vs Git

Collatz Lab is **local-first**: your SQLite database, generated artifacts, exported claim notes, and calibration JSON **do not belong in the repository**. They stay on disk under paths listed in the root `.gitignore`.

## What Git will never overwrite from `git pull`

These paths are **not tracked** (ignored). Remote branches do not contain them, so `git pull` **does not replace** your files there:

- `data/*.db` and related SQLite sidecar files — your lab database and history
- `data/*.json` — e.g. Metal chunk calibration, worker sandbox snapshots
- `data/logs/` — worker / API log files
- `artifacts/` — run JSON, validations, hypothesis evidence files
- `reports/` — generated reports
- `research/claims/` — per-claim markdown exports from your workspace
- `research/paper.json` — dashboard paper bundle generated from the lab
- `data/*.lock` — transient lock files

Your **runs, claims, tasks, and evidence** live in **SQLite** (`data/*.db`). Pulling new **code** does not merge into or wipe that database.

## What `git pull` does update

Only **tracked** paths: source under `backend/`, `dashboard/`, `scripts/`, `docs/`, `research/*.md` (roadmap and notes, **not** `research/claims/`), etc.

## One-time caution after a cleanup commit

If an older commit **used to track** a path (e.g. a file under `research/claims/` or `artifacts/`) and a newer commit **removes** that path from the repo, then **the first `git pull` onto a clone that still had those paths as tracked files** can make Git **remove those paths from the working tree** so it matches the new commit. That is Git behaving normally: “the repo no longer has this file.”

**If you care about the on-disk copies** of files that used to be tracked, back them up before pulling (copy `data/`, `artifacts/`, `research/claims/` elsewhere). Ignored and never-tracked files are not removed by that update.

## Restoring an old tracked file (optional)

If something was deleted from the working tree by a pull but still exists in history:

```bash
git show <commit>:path/to/file > /tmp/recovered
```

## Summary

- **DB + local lab outputs:** ignored → **pull does not overwrite them** going forward.
- **Code and shared docs:** pull updates them.
- **Unsure:** copy `data/` and `research/claims/` before a large pull once; then rely on `.gitignore`.
