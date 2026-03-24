# Roadmap: Metal benchmark in **UI**, **central server**, **hall of fame**

## What works **today** (local only, no UI)

1. On **your Mac**, you run a script that measures **M odd seeds/s** for several Metal chunk sizes.
2. The winner is saved to **`data/metal_sieve_chunk_calibration.json`** next to the project (local SQLite remains your working database).
3. **Local workers** read that file to pick the chunk automatically (if you have not set `COLLATZ_METAL_SIEVE_CHUNK_SIZE`).
4. There is **no** upload to a public server, leaderboard, or dashboard screen yet.

Typical command:

```bash
cd CollatzLab
PYTHONPATH=backend/src ./.venv/bin/python scripts/profile_metal_sieve_chunk.py --quick --reps 5 --write-calibration
# or: npm run bench:metal-chunk
```

Then **restart workers** so they reload the file.

---

## What you may want **later** (clear summary)

| Target | Role |
|--------|------|
| **UI in dashboard** | “Run benchmark” button + progress + result + optional “submit to server”. |
| **Public site + primary DB** | An API (or an “official” Collatz Lab instance) that accepts **anonymized** results (M/s, winning chunk, chip model, RAM, versions). |
| **Local worker** | Runs the same benchmark as today, then **POST** JSON to the server (with consent / API key). |
| **Hall of fame** | Page ranking by **M/s**, **efficiency**, **per chip**, filters (M1 / M2 / …), without exposing personal data. |

This is **new product + infrastructure**; it does not replace the local file until you add explicit sync.

---

## Proposed architecture (phases)

### Phase 1 — **Local UI** (no public server)

- FastAPI endpoint: `POST /api/bench/metal-chunk` (or async job) running the sweep in a **separate thread/process** (benchmark can take minutes).
- Store last result in SQLite: table `metal_benchmark_runs` (timestamp, winner_chunk, m_per_s, interval, parity_ok).
- Dashboard: “Metal benchmark” card + “Write calibration” (same code path as `write_metal_chunk_calibration`).
- **Zero** internet dependency.

### Phase 2 — **Optional telemetry** (your server)

- Same JSON schema as the calibration file + fields: `torch_version`, `metal_helper_git_sha`, `hw_model`, `ram_gb`, `os_build`.
- `POST /api/public/benchmark-results` on the public instance with **rate limit** + schema validation.
- UI consent: “Publish result anonymously”.

### Phase 3 — **Hall of fame**

- SQL aggregations: top per `hw_model`, averages, trends over time (after upgrades).
- Static public page or React route: `/leaderboard/metal-chunk`.

### Phase 4 — **Distributed workers**

- CLI worker: `--submit-benchmark` after a local run or weekly cron.
- Server: weak deduplication (same host + same day) to limit spam.

---

## Data model (minimal server sketch)

```text
benchmark_submission (
  id, created_at,
  benchmark_kind,          -- e.g. metal_sieve_chunk_sweep
  platform,                -- Darwin
  hw_model,                -- MacBookPro18,1 (from sysctl / worker user agent)
  winner_chunk_odds,
  median_odd_per_sec,
  interval_start, interval_end,
  client_version,          -- collatz-lab git tag or py version
  torch_version,
  calibration_json,        -- optional JSONB
  anon_install_id          -- optional stable hash, not email
)
```

---

## Risks / decisions

- **Benchmark on the web server** — not recommended (no user Metal); everything stays **client-side / worker**.
- **Comparability** — M/s depends on `[1,N]` and large `n`; document a **fixed interval** on the leaderboard.
- **Abuse** — without strong auth, results are **indicative**, not “proof”.

---

## Tie-in to current code

- Sweep logic stays in `scripts/profile_metal_sieve_chunk.py` / `collatz_lab` modules; UI will call the same logic or a shared service refactored into `collatz_lab.bench_metal_chunk` (future).
- Local calibration stays in `data/metal_sieve_chunk_calibration.json` until you choose to **sync** from a public DB.

The smallest MVP is **Phase 1** (SQLite table + endpoint + one dashboard card).
